#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "util/eventfd.hh"
#include "util/util.hh"

#include "common.hh"
#include "contextman.hh"

namespace orthrus::compute {

template<typename Model>
class PipedComputeKernel
{
public:
  using ConfigType = typename Model::ConfigType;
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;

  static constexpr KernelType Type = KernelType::SimplePiped;

public:
  template<typename... Args>
  PipedComputeKernel( const NodeConcurrency& concurrency, Args&&... args );

  ~PipedComputeKernel();

  void push( orthrus::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( orthrus::models::BatchedInferenceState<ConfigType>& state );

  EventFD& event_fd() { return event_fd_; }

private:
  using Stage = orthrus::models::InferenceStage;
  using StateType = orthrus::models::BatchedInferenceState<ConfigType>;
  using StatePriorityQueue
    = std::priority_queue<StateQueueItem<ConfigType>, std::deque<StateQueueItem<ConfigType>>, StateCompOp<ConfigType>>;

  template<typename M>
  struct ModelData
  {
    ModelData( std::unique_ptr<M>&& in_model, const NodeConcurrency& in_concurrency );

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
    const NodeConcurrency concurrency;
    const bool can_fuse_to_pre;
    const bool can_fuse_to_cls;

    StatePriorityQueue processing {};
    std::mutex mutex {};
    std::condition_variable cv {};
  };

  // ... -> [pre(a|b) -> att(a|b) -> post(a|b)] * n_layers -> classify(a|b)
  ModelData<Model> model_;

  EventFD event_fd_ {};
  Measurement& __stats__ { global_measurement() };
  std::atomic<bool> running_ { true };

  // <context management>
  // keeping track of the populated contexts for the states
  size_t current_local_state_id_ { 0 };
  std::map<size_t, std::vector<ContextPtr>> context_map_ {};
  std::mutex context_mutex_ {};

  // incoming -> (waiting|{a,b}.processing) -> outgoing
  GlobalQueue<ConfigType> incoming_ {};
  GlobalQueue<ConfigType> waiting_ {};
  GlobalQueue<ConfigType> outgoing_ {};
  // </queues>

  // <threads>
  template<typename M>
  void execution_thread_func( ModelData<M>& model_data );
  void bookkeeping_thread_func();

  std::vector<std::thread> threads_ {};
  // </threads>
};

template<typename Model>
template<typename M>
PipedComputeKernel<Model>::ModelData<M>::ModelData( std::unique_ptr<M>&& in_model,
                                                    const NodeConcurrency& in_concurrency )
  : model( std::move( in_model ) )
  , context_manager( model->settings() )
  , concurrency( in_concurrency )
  , can_fuse_to_pre( concurrency.get( models::InferenceStage::PostAttention )
                     == concurrency.get( models::InferenceStage::PreAttention ) )
  , can_fuse_to_cls( concurrency.get( models::InferenceStage::PostAttention )
                     == concurrency.get( models::InferenceStage::Classification ) )
{
}

template<typename Model>
template<typename... Args>
PipedComputeKernel<Model>::PipedComputeKernel( const NodeConcurrency& concurrency, Args&&... args )
  : model_( std::make_unique<Model>( std::forward<Args>( args )... ), concurrency )
{
  threads_.emplace_back( &PipedComputeKernel::bookkeeping_thread_func, this );
  threads_.emplace_back( &PipedComputeKernel::execution_thread_func<Model>, this, std::ref( model_ ) );
}

template<typename Model>
PipedComputeKernel<Model>::~PipedComputeKernel()
{
  LOG( INFO ) << "PipedComputeKernel shutting down...";
  running_ = false;
  for ( auto& t : threads_ ) {
    t.join();
  }
}

template<typename Model>
void PipedComputeKernel<Model>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  state.set_id( current_local_state_id_++ );

  DLOG( INFO ) << "Pushing state to incoming queue: " << state.debug_string( false );

  {
    std::lock_guard lock { incoming_.mutex };
    incoming_.queue.emplace( std::move( state ) );
  }

  incoming_.cv.notify_one();
}

template<typename Model>
bool PipedComputeKernel<Model>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename Model>
template<typename M>
void PipedComputeKernel<Model>::execution_thread_func(
  typename PipedComputeKernel<Model>::template ModelData<M>& model_data )
{
  while ( running_ ) {
    StateType state {};

    DLOG( WARNING ) << "Current status: " << "incoming_size=" << incoming_.queue.size() << ", "
                    << "processing_size=" << model_.processing.size();

    // get the next state to process
    {
      std::unique_lock lock { model_data.mutex };
      model_data.cv.wait( lock, [&model_data] { return !model_data.processing.empty(); } );
      state = std::move( model_data.processing.top().state );
      model_data.processing.pop();
    }

    DLOG( INFO ) << "Popped state from processing: " << state.debug_string( false );

    const auto local_id = state.id();
    const auto next_stage = state.next_stage();

    // run the corresponding forward function
    switch ( next_stage ) {
      case Stage::PreAttention:
        timeit<IntDistributions::KernelPreAttentionForwardTime>(
          __stats__, [&] { model_data.model->forward_pre_attention( state ); } );
        break;

      case Stage::Attention: {
        std::unique_lock lock { context_mutex_ };
        auto& contexts = context_map_[local_id];
        lock.unlock();
        timeit<IntDistributions::KernelAttentionForwardTime>(
          __stats__, [&] { model_data.model->forward_attention( state, contexts ); } );
      } break;

      case Stage::PostAttention:
        timeit<IntDistributions::KernelPostAttentionForwardTime>( __stats__, [&] {
          model_data.model->forward_post_attention( state, model_data.can_fuse_to_pre, model_data.can_fuse_to_cls );
        } );
        break;

      case Stage::Classification:
        timeit<IntDistributions::KernelClassificationForwardTime>(
          __stats__, [&] { model_data.model->forward_classify( state ); } );
        break;
      default: LOG( FATAL ) << "Invalid stage: " << state.next_stage(); break;
    }

    // We always put the result in outgoing, so TierRouter makes a decision about it.
    {
      // remove the contexts from the context map
      std::lock_guard lock { context_mutex_ };
      context_map_.erase( local_id );
    }

    DLOG( INFO ) << "Pushing state to outgoing queue: " << state.debug_string( false );

    {
      std::lock_guard lock { outgoing_.mutex };
      outgoing_.queue.emplace( std::move( state ) );
    }

    event_fd_.write_event();
  }
}

template<typename Model>
void PipedComputeKernel<Model>::bookkeeping_thread_func()
{
  while ( running_ ) {
    StateType state;

    {
      std::unique_lock lock { incoming_.mutex };
      incoming_.cv.wait( lock, [this] { return !incoming_.queue.empty(); } );

      state = std::move( incoming_.queue.top().state );
      incoming_.queue.pop();

      DLOG( INFO ) << "Popped state from incoming queue: " << state.debug_string( false );
    }

    DCHECK_EQ( model_.concurrency.get( state.next_stage() ), state.batch_size() );

    if ( state.next_stage() == Stage::Attention ) {
      DCHECK( context_map_.find( state.id() ) == context_map_.end() );
      auto contexts_opt = model_.context_manager.get_contexts( state );
      DCHECK( contexts_opt.has_value() )
        << "TierRouter has guaranteed context space, but compute kernel doesn't have enough";

      {
        std::lock_guard lock { context_mutex_ };
        context_map_[state.id()] = contexts_opt.value();
      }
    }

    {
      std::lock_guard lock { model_.mutex };
      model_.processing.emplace( std::move( state ) );
    }

    model_.cv.notify_one();
  }
}

} // namespace orthrus::compute
