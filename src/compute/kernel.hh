#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "models/common/state.hh"
#include "models/types.hh"
#include "monitoring/measurement.hh"
#include "prompt/prompt.hh"
#include "util/eventfd.hh"

#include "common.hh"
#include "contextman.hh"

namespace orthrus::compute {

template<typename Model>
class BatchedComputeKernel
{
public:
  using ModelConfig = typename Model::ConfigType;
  using BatchedState = orthrus::models::BatchedInferenceState<ModelConfig>;
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;

  static constexpr KernelType Type = KernelType::Batched;

private:
  std::unique_ptr<Model> model_;
  std::unique_ptr<PreallocatingContextManager<Model>> context_manager_;

  std::queue<BatchedState> incoming_ {}, waiting_ {}, outgoing_ {};
  std::queue<std::pair<BatchedState, std::vector<ContextPtr>>> processing_ {};

  std::mutex incoming_mutex_ {}, processing_mutex_ {}, outgoing_mutex_ {};
  std::condition_variable_any incoming_cv_ {}, processing_cv_ {};

  EventFD event_fd_ {};
  Measurement& __stats__ { global_measurement() };

  // TODO(): where are stop tokens coming from, and why aren't we using this in other kernels?
  // <threads>
  void execution_thread_func( std::stop_token stoken );
  void bookkeeping_thread_func( std::stop_token stoken );

  std::jthread execution_thread_;
  std::jthread bookkeeping_thread_;
  // </threads>

  void push_to_incoming( BatchedState&& state )
  {
    {
      std::lock_guard lock( incoming_mutex_ );
      incoming_.push( std::move( state ) );
    }

    incoming_cv_.notify_one();
  }

  std::vector<ContextPtr> assemble_contexts( const BatchedState& state )
  {
    auto contexts_opt = context_manager_->get_contexts( state );
    DCHECK( contexts_opt.has_value() )
      << "TierRouter has guaranteed context space, but compute kernel doesn't have enough";
    return contexts_opt.value();
  }

public:
  template<typename... Args>
  BatchedComputeKernel( Args&&... args )
    : model_( std::make_unique<Model>( std::forward<Args>( args )... ) )
    , context_manager_( std::make_unique<PreallocatingContextManager<Model>>( model_->settings() ) )
    , execution_thread_( std::bind( &BatchedComputeKernel::execution_thread_func, this, std::placeholders::_1 ) )
    , bookkeeping_thread_( std::bind( &BatchedComputeKernel::bookkeeping_thread_func, this, std::placeholders::_1 ) )
  {
  }

  void push( BatchedState&& state ) { push_to_incoming( std::move( state ) ); }

  bool pop( BatchedState& state )
  {
    std::lock_guard lock( outgoing_mutex_ );
    if ( outgoing_.empty() )
      return false;
    state = std::move( outgoing_.front() );
    outgoing_.pop();
    return true;
  }

  EventFD& event_fd() { return event_fd_; }

  ~BatchedComputeKernel() { LOG( INFO ) << "BatchedComputeKernel shutting down..."; }
};

template<typename Model>
void BatchedComputeKernel<Model>::execution_thread_func( std::stop_token stoken )
{
  LOG( INFO ) << "BatchedComputeKernel execution thread started.";

  std::pair<BatchedState, std::vector<ContextPtr>> action;

  BatchedState state;
  std::vector<ContextPtr> contexts;

  while ( not stoken.stop_requested() ) {
    {
      std::unique_lock lock( processing_mutex_ );
      if ( not processing_cv_.wait( lock, stoken, [this] { return not processing_.empty(); } ) ) {
        continue; // we were woken up by the stop token
      }

      state = std::move( processing_.front().first );
      // TODO(): why is context moved here?
      contexts = std::move( processing_.front().second );
      processing_.pop();
    }

    timeit<IntDistributions::KernelForwardTime>( __stats__, [&] { model_->forward( state, contexts ); } );

    {
      std::lock_guard lock( outgoing_mutex_ );
      outgoing_.emplace( std::move( state ) );
    }

    event_fd_.write_event();
  }

  LOG( INFO ) << "BatchedComputeKernel execution thread exiting.";
}

template<typename Model>
void BatchedComputeKernel<Model>::bookkeeping_thread_func( std::stop_token stoken )
{
  LOG( INFO ) << "BatchedComputeKernel bookkeeping thread started.";

  BatchedState state;
  std::vector<ContextPtr> contexts;

  while ( not stoken.stop_requested() ) {
    // let's get an action from the incoming_
    {
      std::unique_lock lock( incoming_mutex_ );
      if ( not incoming_cv_.wait( lock, stoken, [this] { return not incoming_.empty(); } ) ) {
        continue;
      }

      state = std::move( incoming_.front() );
      incoming_.pop();
    }

    // let's get the contexts for this state
    contexts = std::move( assemble_contexts( state ) );

    {
      std::lock_guard lock( processing_mutex_ );
      processing_.emplace( std::move( state ), std::move( contexts ) );
    }

    processing_cv_.notify_one();
  }

  LOG( INFO ) << "BatchedComputeKernel bookkeeping thread exiting.";
}

} // namespace orthrus::compute
