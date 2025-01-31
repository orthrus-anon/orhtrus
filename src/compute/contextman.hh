#pragma once

#include <glog/logging.h>
#include <optional>
#include <vector>

#include "models/common/state.hh"
#include "models/types.hh"

namespace orthrus::compute {

/// @brief
/// WARNING: NOT THREAD SAFE
template<typename Model>
class ContextManager
{
private:
  std::unordered_map<orthrus::ContextID, std::shared_ptr<typename Model::ContextType>> contexts_ {};
  const typename Model::SettingsType settings_ {};

public:
  explicit ContextManager( const typename Model::SettingsType& settings )
    : settings_( settings )
  {
  }

  std::shared_ptr<typename Model::ContextType> get_context( const orthrus::ContextID& context_id,
                                                            bool emplace_empty = false )
  {
    auto it = contexts_.find( context_id );
    if ( it != contexts_.end() ) {
      return it->second;
    }

    auto context = std::make_shared<typename Model::ContextType>( settings_ );

    if ( not context.get()->empty() or emplace_empty ) {
      contexts_.emplace( context_id, context );
      DLOG( INFO ) << "(size: " << contexts_.size() << ") Added context for " << context_id;
    }

    return context;
  }

  bool release_context( const orthrus::ContextID& context_id )
  {
    LOG( FATAL ) << "We should never be releasing any context";
    return contexts_.erase( context_id ) > 0;
  }
};

/// @brief
/// WARNING: THREAD SAFE, DO NOT USE ANOTHER LOCK WHILE CALLING
// TODO(): context is guaranteed. Maybe we should stop caring about mapping from context ID to context, and just
//  use an array lookup.
template<typename Model>
class PreallocatingContextManager
{
public:
  using StateType = orthrus::models::BatchedInferenceState<typename Model::ConfigType>;
  using ContextPtr = std::shared_ptr<typename Model::ContextType>;

  explicit PreallocatingContextManager( const typename Model::SettingsType& settings );

  ContextPtr get_context( const ContextID& context_id );

  /// @brief Returns the contexts for all the prompts in the given state. Returns an empty optional if context cannot
  /// be allocated for any of the prompts.
  /// @param state The state for which we want to allocate contexts.
  /// @return An optional containing the contexts for all the prompts in the state, or an empty optional if context
  /// cannot be allocated for some of the prompts.
  std::optional<std::vector<ContextPtr>> get_contexts( const StateType& state, size_t start_ind, size_t end_ind );
  std::optional<std::vector<ContextPtr>> get_contexts( const StateType& state );

  bool release_context( const ContextID& context_id );

  [[nodiscard]] size_t free() const { return free_contexts_.size(); }
  [[nodiscard]] size_t allocated() const { return allocated_contexts_.size(); }
  [[nodiscard]] size_t total() const { return free() + allocated(); }

private:
  std::mutex mutex_ {};

  std::list<ContextPtr> free_contexts_ {};
  const ContextPtr empty_context_;
  std::unordered_map<ContextID, ContextPtr> allocated_contexts_ {};
};

template<typename Model>
PreallocatingContextManager<Model>::PreallocatingContextManager( const typename Model::SettingsType& settings )
  : empty_context_( std::make_shared<typename Model::ContextType>( settings, true ) )
{
  for ( size_t i = 0; i < settings.max_context_count; i++ ) {
    free_contexts_.emplace_back( std::make_shared<typename Model::ContextType>( settings ) );
  }

  const size_t ctx_size = free_contexts_.front()->max_size( settings.num_attention_layers_hosted() );
  LOG( INFO ) << "Preallocated " << settings.max_context_count << " contexts (" << settings.max_context_count << "x"
              << ( ctx_size >> 20 ) << "MiB=" << ( settings.max_context_count * ctx_size >> 20 ) << " MiB,"
              << typeid( *this ).name() << ")";
}

template<typename Model>
typename PreallocatingContextManager<Model>::ContextPtr PreallocatingContextManager<Model>::get_context(
  const ContextID& context_id )
{
  // Deprecated in favor of get_contexts( const StateType& state, size_t start_ind, size_t end_ind )
  std::lock_guard lock { mutex_ };

  auto it = allocated_contexts_.find( context_id );
  if ( it != allocated_contexts_.end() ) {
    return it->second;
  }

  if ( free_contexts_.empty() ) {
    LOG( WARNING ) << "No free contexts available.";
    return {};
  }

  auto& ctx = free_contexts_.front();
  auto [it_new, inserted] = allocated_contexts_.emplace( context_id, std::move( ctx ) );
  free_contexts_.pop_front();
  return it_new->second;
}

template<typename Model>
std::optional<std::vector<typename PreallocatingContextManager<Model>::ContextPtr>>
PreallocatingContextManager<Model>::get_contexts( const StateType& state, const size_t start_ind, const size_t end_ind )
{
  size_t no_context_count = 0;
  std::vector<ContextPtr> contexts;
  contexts.reserve( end_ind - start_ind );

  std::lock_guard lock { mutex_ };

  // TODO: can probably avoid two map lookups

  // (1) let's make sure we can assign contexts to all the prompts first
  for ( size_t i = start_ind; i < end_ind; i++ ) {
    if ( state.active( i ) ) {
      const auto it = allocated_contexts_.find( state.context_id( i ) );
      if ( it == allocated_contexts_.end() ) {
        no_context_count++;
      }
    }
  }

  if ( no_context_count > free() ) {
    // not enough free contexts to allocate
    return std::nullopt;
  }

  // (2) now we can assign the contexts
  for ( size_t i = start_ind; i < end_ind; i++ ) {
    if ( not state.active( i ) ) {
      contexts.push_back( empty_context_ );
    } else {
      const auto it = allocated_contexts_.find( state.context_id( i ) );
      if ( it == allocated_contexts_.end() ) {
        auto& ctx = free_contexts_.front();
        auto [it_new, inserted] = allocated_contexts_.emplace( state.context_id( i ), std::move( ctx ) );
        contexts.push_back( it_new->second );
        free_contexts_.pop_front();
      } else {
        contexts.push_back( it->second );
      }
    }
  }

  return contexts;
}

template<typename Model>
std::optional<std::vector<typename PreallocatingContextManager<Model>::ContextPtr>>
PreallocatingContextManager<Model>::get_contexts( const StateType& state )
{
  return get_contexts( state, 0, state.batch_size() );
}

template<typename Model>
bool PreallocatingContextManager<Model>::release_context( const ContextID& context_id )
{
  LOG( FATAL ) << "We should never be releasing contexts";
  std::lock_guard lock { mutex_ };

  auto it = allocated_contexts_.find( context_id );
  if ( it == allocated_contexts_.end() ) {
    // NOTE(): I don't like this behavior (silently ignoring the release of a non-allocated context), but let's
    // keep it for now.
    return false;
  }

  free_contexts_.push_back( std::move( it->second ) );
  allocated_contexts_.erase( it );
  return true;
}

}
