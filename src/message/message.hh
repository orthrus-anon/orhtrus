#pragma once

#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <string_view>

#include "handler.hh"
#include "net/session.hh"

namespace orthrus::core {

class Message
{
public:
  enum class OpCode : uint8_t
  {
    Hey = 0x1,
    Ping,
    Bye,

    InitializeWorker,
    AckInitialize,
    SetRoute,
    AckRoute,
    PushDummyPrompts,
    PushPrompts,
    PushCompletions,
    BatchedInferenceState,

    __COUNT
  };

  static constexpr char const* OPCODE_NAMES[static_cast<int>( OpCode::__COUNT )] = {
    "", // OpCode 0x0 is not used

    "Hey",
    "Ping",
    "Bye",

    "InitializeWorker",
    "AckInitialize",
    "SetRoute",
    "AckRoute",
    "PushDummyPrompts",
    "PushPrompts",
    "PushCompletions",
    "BatchedInferenceState",
  };

  constexpr static size_t HEADER_LENGTH = 5;

private:
  uint32_t payload_length_ { 0 };
  OpCode opcode_ { OpCode::Hey };
  std::string payload_ {};

public:
  Message( const std::string_view& header, std::string&& payload );
  Message( const OpCode opcode, std::string&& payload );

  uint32_t payload_length() const { return payload_length_; }
  OpCode opcode() const { return opcode_; }
  const std::string& payload() const { return payload_; }

  void serialize_header( std::string& output );

  size_t total_length() const { return HEADER_LENGTH + payload_length(); }
  static uint32_t expected_payload_length( const std::string_view header );

  std::string info() const;
};

class MessageParser
{
private:
  std::optional<size_t> expected_payload_length_ { std::nullopt };

  std::string incomplete_header_ {};
  std::string incomplete_payload_ {};

  std::queue<Message> completed_messages_ {};

  void complete_message();

public:
  size_t parse( const std::string_view buf );

  bool empty() const { return completed_messages_.empty(); }
  Message& front() { return completed_messages_.front(); }
  void pop() { completed_messages_.pop(); }

  size_t size() const { return completed_messages_.size(); }
};

template<class SessionType>
class MessageHandler : public orthrus::MessageHandler<SessionType, Message, Message>
{
protected:
  std::queue<Message> outgoing_ {};
  MessageParser incoming_ {};

  std::string current_outgoing_header_ {};
  std::string_view current_outgoing_unsent_header_ {};
  std::string_view current_outgoing_unsent_payload_ {};

  void load();

  bool outgoing_empty() override;
  bool incoming_empty() const override { return incoming_.empty(); }
  Message& incoming_front() override { return incoming_.front(); }
  void incoming_pop() override { incoming_.pop(); }

  void write( RingBuffer& out ) override;
  void read( RingBuffer& in ) override;

public:
  using orthrus::MessageHandler<SessionType, Message, Message>::MessageHandler;

  ~MessageHandler() {}

  void push_message( Message&& msg ) override;
};

template<class SessionType>
class DelayedMessageHandler : public MessageHandler<SessionType>
{
private:
  using Clock = std::chrono::steady_clock;

  mutable std::queue<std::pair<Clock::time_point, Message>> delayed_messages_ {};
  std::chrono::milliseconds delay_ {};

protected:
  bool outgoing_empty() override
  {
    if ( not MessageHandler<SessionType>::outgoing_empty() ) {
      /* already have stuff to sent in its queue */
      // NOTE(): doing this is not necessary, but I want to avoid calling Clock::now() if we don't need to.
      return false;
    }

    // are the any delayed messages we can send now?
    while ( not delayed_messages_.empty() and delayed_messages_.front().first <= Clock::now() ) {
      MessageHandler<SessionType>::push_message( std::move( delayed_messages_.front().second ) );
      delayed_messages_.pop();
    }

    return MessageHandler<SessionType>::outgoing_empty();
  }

public:
  using MessageHandler<SessionType>::MessageHandler;

  DelayedMessageHandler( SessionType&& session, const std::chrono::milliseconds delay )
    : MessageHandler<SessionType>( std::move( session ) )
  {
    delay_ = delay;
  }

  void push_message( Message&& msg ) override { delayed_messages_.emplace( Clock::now() + delay_, std::move( msg ) ); }

  std::chrono::milliseconds time_to_next_message() const
  {
    if ( delayed_messages_.empty() ) {
      return delay_;
    }

    return std::max(
      std::chrono::milliseconds { 0 },
      std::chrono::duration_cast<std::chrono::milliseconds>( delayed_messages_.front().first - Clock::now() ) );
  }
};

} // namespace orthrus::core
