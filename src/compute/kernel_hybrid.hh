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

template<typename ModelA, typename ModelB>
requires std::same_as<typename ModelA::ConfigType, typename ModelB::ConfigType>
class HybridComputeKernel
{
public:
  using ConfigType = typename ModelA::ConfigType;
  using ContextPtrA = std::shared_ptr<typename ModelA::ContextType>;
  using ContextPtrB = std::shared_ptr<typename ModelB::ContextType>;

  static constexpr KernelType Type = KernelType::Hybrid;

public:
  template<typename... Args>
  HybridComputeKernel( const NodeConcurrency& concurrency_a, const NodeConcurrency& concurrency_b, Args&&... args );

  ~HybridComputeKernel();

  void push( orthrus::models::BatchedInferenceState<ConfigType>&& state );
  bool pop( orthrus::models::BatchedInferenceState<ConfigType>& state );

  EventFD& event_fd() { return event_fd_; }

private:
  using Stage = orthrus::models::InferenceStage;
  using StateType = orthrus::models::BatchedInferenceState<ConfigType>;
  using StatePriorityQueue
    = std::priority_queue<StateQueueItem<ConfigType>, std::deque<StateQueueItem<ConfigType>>, StateCompOp<ConfigType>>;

  template<typename M>
  requires std::same_as<M, ModelA> || std::same_as<M, ModelB>
  struct ModelData
  {
    ModelData( std::unique_ptr<M>&& in_model, const NodeConcurrency& in_concurrency );

    std::unique_ptr<M> model;
    PreallocatingContextManager<M> context_manager;
    const NodeConcurrency concurrency;

    StatePriorityQueue processing {};
    std::mutex mutex {};
    std::condition_variable cv {};
  };

  // ... -> [pre(a|b) -> att(a|b) -> post(a|b)] * n_layers -> classify(a|b)
  ModelData<ModelA> a_;
  ModelData<ModelB> b_;

  EventFD event_fd_ {};
  std::atomic<bool> running_ { true };

  // keeping track of splitted states and merge them back when needed
  size_t current_local_state_id_ { 0 };
  std::map<size_t, std::pair<std::optional<StateType>, std::optional<StateType>>> splitted_state_map_ {};
  std::mutex splitted_state_mutex_ {};

  // <context management>
  // keeping track of the populated contexts for the states
  std::map<size_t, std::pair<std::vector<ContextPtrA>, std::vector<ContextPtrB>>> context_map_ {};
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

template<typename ModelA, typename ModelB>
template<typename M>
HybridComputeKernel<ModelA, ModelB>::ModelData<M>::ModelData( std::unique_ptr<M>&& in_model,
                                                              const NodeConcurrency& in_concurrency )
  : model( std::move( in_model ) )
  , context_manager( model->settings() )
  , concurrency( in_concurrency )
{
}

template<typename ModelA, typename ModelB>
template<typename... Args>
HybridComputeKernel<ModelA, ModelB>::HybridComputeKernel( const NodeConcurrency& concurrency_a,
                                                          const NodeConcurrency& concurrency_b,
                                                          Args&&... args )
  : a_( std::make_unique<ModelA>( std::forward<Args>( args )... ), concurrency_a )
  , b_( std::make_unique<ModelB>( std::forward<Args>( args )... ), concurrency_b )
{
  threads_.emplace_back( &HybridComputeKernel::bookkeeping_thread_func, this );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelA>, this, std::ref( a_ ) );
  threads_.emplace_back( &HybridComputeKernel::execution_thread_func<ModelB>, this, std::ref( b_ ) );
}

template<typename ModelA, typename ModelB>
HybridComputeKernel<ModelA, ModelB>::~HybridComputeKernel()
{
  LOG( INFO ) << "HybridComputeKernel shutting down...";
  running_ = false;
  for ( auto& t : threads_ ) {
    t.join();
  }
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::push( models::BatchedInferenceState<ConfigType>&& state )
{
  state.set_id( current_local_state_id_++ );

  DLOG( INFO ) << "Pushing state to incoming queue: " << state.debug_string( false );

  {
    std::lock_guard lock { incoming_.mutex };
    incoming_.queue.emplace( std::move( state ) );
  }

  incoming_.cv.notify_one();
}

template<typename ModelA, typename ModelB>
bool HybridComputeKernel<ModelA, ModelB>::pop( models::BatchedInferenceState<ConfigType>& state )
{
  std::lock_guard lock { outgoing_.mutex };

  if ( outgoing_.queue.empty() ) {
    return false;
  }

  state = std::move( outgoing_.queue.top().state );
  outgoing_.queue.pop();
  return true;
}

template<typename ModelA, typename ModelB>
template<typename M>
void HybridComputeKernel<ModelA, ModelB>::execution_thread_func(
  typename HybridComputeKernel<ModelA, ModelB>::template ModelData<M>& model_data )
{
  while ( running_ ) {
    StateType state {};

    DLOG( WARNING ) << "Current status: " << "incoming_size=" << incoming_.queue.size() << ", "
                    << "waiting_size=" << waiting_.queue.size() << ", " << "a_processing_size=" << a_.processing.size()
                    << ", " << "b_processing_size=" << b_.processing.size();

    // get the next state to process
    {
      std::unique_lock lock { model_data.mutex };
      model_data.cv.wait( lock, [&model_data] { return !model_data.processing.empty(); } );
      state = std::move( model_data.processing.top().state );
      model_data.processing.pop();
    }

    DLOG( INFO ) << "Popped state from processing: " << state.debug_string( false ) << " (by "
                 << ( std::is_same_v<ModelA, M> ? "A" : "B" ) << ")";

    const auto local_id = state.id();
    const auto next_stage = state.next_stage();

    // run the corresponding forward function
    switch ( next_stage ) {
      case Stage::PreAttention: model_data.model->forward_pre_attention( state ); break;

      case Stage::Attention: {
        std::unique_lock lock { context_mutex_ };

        if constexpr ( std::same_as<M, ModelA> ) {
          auto& contexts = context_map_[local_id].first;
          lock.unlock();
          model_data.model->forward_attention( state, contexts );
        } else {
          auto& contexts = context_map_[local_id].second;
          lock.unlock();
          model_data.model->forward_attention( state, contexts );
        }
      } break;

      case Stage::PostAttention: model_data.model->forward_post_attention( state ); break;
      case Stage::Classification: model_data.model->forward_classify( state ); break;
      default: LOG( FATAL ) << "Invalid stage: " << state.next_stage(); break;
    }

    std::optional<StateType> merged_state;
    {
      std::lock_guard lock { splitted_state_mutex_ };

      auto& [state_a, state_b] = splitted_state_map_[local_id];

      if constexpr ( std::same_as<M, ModelA> ) {
        state_a.emplace( std::move( state ) );
      } else {
        state_b.emplace( std::move( state ) );
      }

      if ( state_a.has_value() and state_b.has_value() ) {
        // merge back the states
        if ( state_b->empty() ) {
          merged_state.emplace( std::move( *state_a ) );
        } else if ( state_a->empty() ) {
          merged_state.emplace( std::move( *state_b ) );
        } else {
          state_a->merge( std::move( *state_b ) );
          merged_state.emplace( std::move( *state_a ) );
        }
        splitted_state_map_.erase( local_id );
      }
    }

    if ( merged_state.has_value() ) {
      // we need to send it to the outgoing queue regardless of what it is
      {
        // remove the contexts from the context map
        std::unique_lock lock { context_mutex_ };
        context_map_.erase( local_id );
      }

      DLOG( INFO ) << "Pushing state to outgoing queue: " << merged_state->debug_string( false );

      {
        std::lock_guard lock { outgoing_.mutex };
        outgoing_.queue.emplace( std::move( *merged_state ) );
      }

      event_fd_.write_event();
    }
  }
}

template<typename ModelA, typename ModelB>
void HybridComputeKernel<ModelA, ModelB>::bookkeeping_thread_func()
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

    DCHECK_EQ( a_.concurrency.get( state.next_stage() ) + b_.concurrency.get( state.next_stage() ),
               state.batch_size() );

    // can we, or have we already, allocated the contexts for this state?

    if ( state.next_stage() == Stage::Attention ) {
      DCHECK( context_map_.find( state.id() ) == context_map_.end() );

      auto contexts_a_opt = a_.context_manager.get_contexts( state, 0, a_.concurrency.get( Stage::Attention ) );
      auto contexts_b_opt
        = b_.context_manager.get_contexts( state, a_.concurrency.get( Stage::Attention ), state.batch_size() );

      DCHECK( contexts_a_opt.has_value() )
        << "TierRouter has guaranteed context space, but compute kernel doesn't have enough in A";
      DCHECK( contexts_b_opt.has_value() )
        << "TierRouter has guaranteed context space, but compute kernel doesn't have enough in B";

      {
        std::lock_guard lock { context_mutex_ };
        context_map_[state.id()] = std::make_pair( contexts_a_opt.value(), contexts_b_opt.value() );
      }
    }

    const auto next_stage = state.next_stage();
    const auto next_layer = state.next_layer();

    // do we need to split this state?
    if ( a_.concurrency.get( next_stage ) > 0 && b_.concurrency.get( next_stage ) > 0 ) {
      // split the state
      auto [state_a, state_b] = state.split( a_.concurrency.get( next_stage ) );

      DCHECK_EQ( state_a.batch_size(), a_.concurrency.get( next_stage ) );
      DCHECK_EQ( state_b.batch_size(), b_.concurrency.get( next_stage ) );

      {
        std::lock_guard lock { a_.mutex };
        a_.processing.emplace( std::move( state_a ) );
      }
      a_.cv.notify_one();

      {
        std::lock_guard lock { b_.mutex };
        b_.processing.emplace( std::move( state_b ) );
      }
      b_.cv.notify_one();
    } else {
      if ( a_.concurrency.get( next_stage ) == state.batch_size() ) {
        DLOG( INFO ) << "Pushing state to A's processing queue: " << state.debug_string( false );

        {
          std::lock_guard lock { splitted_state_mutex_ };
          splitted_state_map_[state.id()].second.emplace();
        }

        {
          std::lock_guard lock { a_.mutex };
          a_.processing.emplace( std::move( state ) );
        }
        a_.cv.notify_one();
      } else if ( b_.concurrency.get( next_stage ) == state.batch_size() ) {
        DLOG( INFO ) << "Pushing state to B's processing queue: " << state.debug_string( false );

        {
          std::lock_guard lock { splitted_state_mutex_ };
          splitted_state_map_[state.id()].first.emplace();
        }

        {
          std::lock_guard lock { b_.mutex };
          b_.processing.emplace( std::move( state ) );
        }
        b_.cv.notify_one();
      }
    }
  }
}

} // namespace orthrus::compute
