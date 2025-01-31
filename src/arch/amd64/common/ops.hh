#pragma once

#include <memory>
#include <random>

#include "arch/float.hh"
#include "models/common/ops/concept.hh"
#include "util/random.hh"

namespace orthrus::models::common::amd64 {

template<typename DType>
class Operations
{
public:
  using DeviceUniquePtr = std::unique_ptr<DType>;

public:
  Operations() {}
  ~Operations() {}

  Operations( const Operations& ) = delete;
  Operations& operator=( const Operations& ) = delete;
  Operations( Operations&& ) = default;
  Operations& operator=( Operations&& ) = default;

  template<uint64_t size>
  void accum( DType* a, const DType* b, const uint64_t batch_size ) const;

  template<uint64_t size>
  void rmsnorm( DType* o, const DType* x, orthrus::float32_t* temp, const DType* weight, const uint64_t batch_size )
    const;

  template<uint64_t n>
  void argmax( uint32_t* output, const DType* v, DType* temp, const uint64_t batch_size ) const;

  template<uint64_t hidden_dim>
  void silu( DType* hb, DType* hb2, const uint64_t batch_size ) const;

  template<uint64_t s, uint64_t r>
  void matmul( DType* xo, const DType* x, const DType* w, const uint64_t b ) const;

  void untemplated_matmul( DType* xo,
                           const DType* x,
                           const DType* w,
                           const uint64_t s,
                           const uint64_t r,
                           const uint64_t b ) const;

  void soft_sample( DType* v, const std::vector<float>& temperatures, const size_t vocab_size ) const;

  DeviceUniquePtr device_allocate( const uint64_t size_bytes ) const;

  void randomize_device_buffer( DType* buffer, const uint64_t len, const float min, const float max ) const;

  void copy( DType* dst,
             const DType* src,
             const uint64_t len_bytes,
             const CopyType type,
             const bool async = false ) const;

  void print( const DType* x, const uint64_t b, const std::string base ) const;
};

static_assert( OperationsConcept<Operations<orthrus::float32_t>, orthrus::float32_t> );
static_assert( OperationsConcept<Operations<orthrus::float16_t>, orthrus::float16_t> );
static_assert( OperationsConcept<Operations<orthrus::bfloat16_t>, orthrus::bfloat16_t> );

// helper functions are in this anonymous namespace
namespace {

namespace { // matmul

template<typename DType, uint64_t m, uint64_t k, uint64_t lda, uint64_t ldb, uint64_t ldc>
void fast_matmul_row_major( uint64_t n, const DType* A, const DType* B, DType* C )
{
  uint64_t row;
  uint64_t col;
#pragma omp parallel for private( row, col ) shared( A, B, C ) collapse( 2 )
  for ( row = 0; row < m; row++ ) {
    for ( col = 0; col < n; col++ ) {
      orthrus::float32_t sum = 0.0;

      for ( uint64_t p = 0; p < k; ++p ) {
        const orthrus::float32_t a_value = A[row * lda + p];
        const orthrus::float32_t b_value = B[col * ldb + p];
        sum += a_value * b_value;
      }

      C[col * ldc + row] = DType( sum );
    }
  }
}

}

namespace { // soft_sample

template<typename DType>
void gumbel_fix( DType* array, orthrus::float32_t temp, const size_t vocab_size )
{
  for ( uint64_t i = 0; i < vocab_size; i++ ) {
    orthrus::float32_t myrandf = static_cast<orthrus::float32_t>( rand() ) / RAND_MAX;
    myrandf = logf( -logf( myrandf ) );
    array[i] = static_cast<DType>( static_cast<orthrus::float32_t>( array[i] ) / temp - myrandf );
  }
}

}

} // end of anonymous namespace for helper functions

template<typename DType>
void Operations<DType>::print( const DType* x, const uint64_t b, const std::string base ) const
{
  printf( "%s", base.c_str() );
  for ( uint64_t i = 0; i < b; i++ ) {
    orthrus::float32_t c = static_cast<orthrus::float32_t>( x[i] );
    printf( "\t%.10f", c );
  }
  printf( "\n" );
}

template<typename DType>
template<uint64_t size>
void Operations<DType>::accum( DType* a, const DType* b, const uint64_t batch_size ) const
{
  uint64_t b_idx;
  uint64_t i;
#pragma omp parallel for private( b_idx, i ) collapse( 2 )
  for ( b_idx = 0; b_idx < batch_size; b_idx++ ) {
    for ( i = 0; i < size; i++ ) {
      a[b_idx * size + i] = static_cast<DType>( static_cast<orthrus::float32_t>( a[b_idx * size + i] )
                                                + static_cast<orthrus::float32_t>( b[b_idx * size + i] ) );
    }
  }
}

template<typename DType>
template<uint64_t size>
void Operations<DType>::rmsnorm( DType* output,
                                 const DType* x,
                                 orthrus::float32_t*,
                                 const DType* weight,
                                 const uint64_t batch_size ) const
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    const DType* X = x + b * size;
    DType* O = output + b * size;

    // calculate sum of squares
    orthrus::float32_t ss = 0.0f;
    for ( uint64_t j = 0; j < size; j++ ) {
      ss += static_cast<orthrus::float32_t>( X[j] ) * static_cast<orthrus::float32_t>( X[j] );
    }

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf( ss );

    // normalize and scale
    for ( uint64_t j = 0; j < size; j++ ) {
      O[j] = static_cast<DType>( static_cast<orthrus::float32_t>( weight[j] )
                                 * ( ss * static_cast<orthrus::float32_t>( X[j] ) ) );
    }
  }
}

template<typename DType>
template<uint64_t n>
void Operations<DType>::argmax( uint32_t* output, const DType* v, DType*, const uint64_t batch_size ) const
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    const DType* this_v = v + b * n;

    uint64_t max_i = 0;
    DType max_p = this_v[0];

    for ( uint64_t i = 1; i < n; i++ ) {
      if ( this_v[i] > max_p ) {
        max_i = i;
        max_p = this_v[i];
      }
    }

    output[b] = max_i;
  }
}

template<typename DType>
template<uint64_t hidden_dim>
void Operations<DType>::silu( DType* hb, DType* hb2, const uint64_t batch_size ) const
{
  uint64_t b;
#pragma omp parallel for private( b )
  for ( b = 0; b < batch_size; b++ ) {
    DType* current_hb = hb + b * hidden_dim;
    DType* current_hb2 = hb2 + b * hidden_dim;

    for ( size_t i = 0; i < hidden_dim; i++ ) {
      const orthrus::float32_t x = static_cast<orthrus::float32_t>( current_hb[i] );
      const orthrus::float32_t y = static_cast<orthrus::float32_t>( current_hb2[i] );
      current_hb[i] = static_cast<DType>( x / ( 1.0f + expf( -x ) ) * y );
    }
  }
}

template<typename DType>
template<uint64_t s, uint64_t r>
void Operations<DType>::matmul( DType* xout, const DType* x, const DType* w, const uint64_t b ) const
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)

  constexpr uint64_t m = r;
  constexpr uint64_t k = s;
  const uint64_t n = b;
  constexpr uint64_t lda = k;
  constexpr uint64_t ldb = k;
  constexpr uint64_t ldc = m;

  fast_matmul_row_major<DType, m, k, lda, ldb, ldc>( n, w, x, xout );
}

template<typename DType>
void Operations<DType>::untemplated_matmul( DType* xout,
                                            const DType* x,
                                            const DType* w,
                                            const uint64_t s,
                                            const uint64_t r,
                                            const uint64_t b ) const
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)

  const uint64_t m = r;
  const uint64_t k = s;
  const uint64_t n = b;
  const uint64_t lda = k;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  const DType* A = w;
  const DType* B = x;
  DType* C = xout;

  uint64_t row;
  uint64_t col;
#pragma omp parallel for private( row, col ) shared( A, B, C ) collapse( 2 )
  for ( row = 0; row < m; row++ ) {
    for ( col = 0; col < n; col++ ) {
      orthrus::float32_t sum = 0.0;

      for ( uint64_t p = 0; p < k; ++p ) {
        const orthrus::float32_t a_value = A[row * lda + p];
        const orthrus::float32_t b_value = B[col * ldb + p];
        sum += a_value * b_value;
      }

      C[col * ldc + row] = DType( sum );
    }
  }
}

template<typename DType>
void Operations<DType>::soft_sample( DType* v, const std::vector<float>& temperatures, const size_t vocab_size ) const
{
  uint64_t i;
#pragma omp parallel for private( i )
  for ( i = 0; i < temperatures.size(); i++ ) {
    if ( temperatures[i] > 0 ) {
      gumbel_fix<DType>( v + i * vocab_size, temperatures[i], vocab_size );
    }
  }
}

template<typename DType>
Operations<DType>::DeviceUniquePtr Operations<DType>::device_allocate( const uint64_t size ) const
{
  return DeviceUniquePtr { reinterpret_cast<DType*>( new uint8_t[size] ) };
}

template<typename DType>
void Operations<DType>::copy( DType* dst, const DType* src, const uint64_t len_bytes, const CopyType, const bool ) const
{
  std::memcpy( dst, src, len_bytes );
}

template<typename DType>
void Operations<DType>::randomize_device_buffer( DType* buffer,
                                                 const uint64_t len,
                                                 const float min,
                                                 const float max ) const
{
  util::randomize_buffer( buffer, len, min, max );
}

} // namespace orthrus::models::common::amd64
