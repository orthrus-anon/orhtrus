#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace orthrus::models::llama2 {

template<class T>
concept ModelConfig = requires( T t ) {
  requires std::is_unsigned_v<decltype( T::dim )>;
  requires std::is_unsigned_v<decltype( T::kv_dim )>;
  requires std::is_unsigned_v<decltype( T::hidden_dim )>;
  requires std::is_unsigned_v<decltype( T::n_layers )>;
  requires std::is_unsigned_v<decltype( T::head_size )>;
  requires std::is_unsigned_v<decltype( T::n_heads )>;
  requires std::is_unsigned_v<decltype( T::n_kv_heads )>;
  requires std::is_unsigned_v<decltype( T::gqa_size )>;
  requires std::is_unsigned_v<decltype( T::vocab_size )>;
  requires std::is_unsigned_v<decltype( T::seq_len )>;
  requires std::is_convertible_v<decltype( T::wcls_present ), bool>;
  requires std::is_unsigned_v<decltype( T::attention_rounds )>;

  // Special tokens
  requires std::is_unsigned_v<decltype( T::token_bos )>;
  requires std::is_unsigned_v<decltype( T::token_eos )>;
  requires std::is_unsigned_v<decltype( T::token_eot )>;

  // Derived constants
  requires T::gqa_size == T::n_heads / T::n_kv_heads;
  requires T::kv_dim == T::dim / T::gqa_size;
  requires T::head_size == T::dim / T::n_heads;
};

namespace configs {

struct Llama2
{
  constexpr static uint64_t vocab_size = 32'000;
  constexpr static uint32_t token_bos = 1; // beginning of sentence
  constexpr static uint32_t token_eos = 2; // end of sentence
  constexpr static uint32_t token_eot = 2; // end of turn
};

struct Llama3
{
  constexpr static uint64_t vocab_size = 128'256;
  constexpr static uint32_t token_bos = 128'000;
  constexpr static uint32_t token_eos = 128'001;
  constexpr static uint32_t token_eot = 128'009;
};

struct Llama3_405B : public Llama3
{
  constexpr static uint64_t dim = 16384;
  constexpr static uint64_t kv_dim = 1024;
  constexpr static uint64_t hidden_dim = 53248;
  constexpr static uint64_t n_layers = 126;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 128;
  constexpr static uint64_t n_kv_heads = 8;
  constexpr static uint64_t gqa_size = 16;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 1;
};

struct Llama3_70B : public Llama3
{
  constexpr static uint64_t dim = 8192;
  constexpr static uint64_t kv_dim = 1024;
  constexpr static uint64_t hidden_dim = 28672;
  constexpr static uint64_t n_layers = 80;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 64;
  constexpr static uint64_t n_kv_heads = 8;
  constexpr static uint64_t gqa_size = 8;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 1;
};

struct Llama3_8B : public Llama3
{
  constexpr static uint64_t dim = 4096;
  constexpr static uint64_t kv_dim = 1024;
  constexpr static uint64_t hidden_dim = 14336;
  constexpr static uint64_t n_layers = 32;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 32;
  constexpr static uint64_t n_kv_heads = 8;
  constexpr static uint64_t gqa_size = 4;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 1;
};

struct Llama2_70B_Chat : public Llama2
{
  constexpr static uint64_t dim = 8192;
  constexpr static uint64_t kv_dim = 1024;
  constexpr static uint64_t hidden_dim = 28672;
  constexpr static uint64_t n_layers = 80;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 64;
  constexpr static uint64_t n_kv_heads = 8;
  constexpr static uint64_t gqa_size = 8;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 1;
};

struct Llama2_70B_Chat_4K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 4096;
};

struct Llama2_70B_Chat_8K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 8192;
};

struct Llama2_70B_Chat_16K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 16384;
};

struct Llama2_70B_Chat_32K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 32768;
};

struct Llama2_70B_Chat_64K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 65536;
};

struct Llama2_70B_Chat_128K : public Llama2_70B_Chat
{
  constexpr static uint64_t seq_len = 131072;
};

struct Llama2_13B_Chat : public Llama2
{
  constexpr static uint64_t dim = 5120;
  constexpr static uint64_t kv_dim = 5120;
  constexpr static uint64_t hidden_dim = 13824;
  constexpr static uint64_t n_layers = 40;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 40;
  constexpr static uint64_t n_kv_heads = 40;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 2;
};

struct Llama2_7B_Chat : public Llama2
{
  constexpr static uint64_t dim = 4096;
  constexpr static uint64_t kv_dim = 4096;
  constexpr static uint64_t hidden_dim = 11008;
  constexpr static uint64_t n_layers = 32;
  constexpr static uint64_t head_size = 128;
  constexpr static uint64_t n_heads = 32;
  constexpr static uint64_t n_kv_heads = 32;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t seq_len = 2048;
  constexpr static bool wcls_present = true;
  constexpr static uint64_t attention_rounds = 2;
};

struct Stories_110M : public Llama2
{
  constexpr static uint64_t dim = 768;
  constexpr static uint64_t kv_dim = 768;
  constexpr static uint64_t hidden_dim = 2048;
  constexpr static uint64_t n_layers = 12;
  constexpr static uint64_t head_size = 64;
  constexpr static uint64_t n_heads = 12;
  constexpr static uint64_t n_kv_heads = 12;
  constexpr static uint64_t gqa_size = 1;
  constexpr static uint64_t seq_len = 1024;
  constexpr static bool wcls_present = false;
  constexpr static uint64_t attention_rounds = 4;
};

static_assert( ModelConfig<Llama3_405B> );
static_assert( ModelConfig<Llama3_70B> );
static_assert( ModelConfig<Llama3_8B> );
static_assert( ModelConfig<Llama2_70B_Chat> );
static_assert( ModelConfig<Llama2_70B_Chat_4K> );
static_assert( ModelConfig<Llama2_70B_Chat_8K> );
static_assert( ModelConfig<Llama2_70B_Chat_16K> );
static_assert( ModelConfig<Llama2_70B_Chat_32K> );
static_assert( ModelConfig<Llama2_70B_Chat_64K> );
static_assert( ModelConfig<Llama2_70B_Chat_128K> );
static_assert( ModelConfig<Llama2_13B_Chat> );
static_assert( ModelConfig<Llama2_7B_Chat> );
static_assert( ModelConfig<Stories_110M> );

} // namespace configs

} // namespace orthrus::models::llama2
