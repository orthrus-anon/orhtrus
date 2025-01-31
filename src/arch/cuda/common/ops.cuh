#pragma once

#include <glog/logging.h>
#include <random>
#include <source_location>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "arch/float.hh"
#include "models/common/ops/concept.hh"

#include "util/random.hh"

namespace orthrus::models::common::cuda {

constexpr size_t TPB = 64;         /* threads per block */
constexpr size_t NRBS = 32;        /* norm reduce block size */
constexpr size_t AMRBS = 128;      /* argmax reduce block size */
constexpr size_t LPK = 64;         /* gumbel_fix loop iterations per kernel */
constexpr size_t MAX_STREAMS = 64; /* max streams in cuda/cublas */

void CHECK_CUBLAS( const cublasStatus_t err, const std::source_location l = std::source_location::current() );
void CHECK_CUDA( const cudaError_t err, const std::source_location l = std::source_location::current() );

template<typename PtrType>
struct CUDADeleter
{
  void operator()( PtrType* ptr ) const
  {
    if ( ptr )
      cudaFree( ptr );
  }
};

template<bool flag = false>
void STATIC_ASSERT_NO_MATCH()
{
  static_assert( flag, "Unsupported type" );
}

template<typename DType>
__host__ __device__ cudaDataType_t get_cuda_data_type()
{
  if constexpr ( std::is_same_v<DType, orthrus::float16_t> ) {
    return CUDA_R_16F;
  } else if constexpr ( std::is_same_v<DType, orthrus::bfloat16_t> ) {
    return CUDA_R_16BF;
  } else if constexpr ( std::is_same_v<DType, orthrus::float32_t> ) {
    return CUDA_R_32F;
  } else {
    STATIC_ASSERT_NO_MATCH();
  }
}

template<typename DType>
__host__ __device__ DType get_dtype_infinity()
{
  if constexpr ( std::is_same_v<DType, orthrus::float16_t> ) {
    return CUDART_INF_FP16;
  } else if constexpr ( std::is_same_v<DType, orthrus::bfloat16_t> ) {
    return CUDART_INF_BF16;
  } else if constexpr ( std::is_same_v<DType, orthrus::float32_t> ) {
    return INFINITY;
  } else {
    STATIC_ASSERT_NO_MATCH();
  }
}

template<typename DType>
class Operations
{
public:
  using DeviceUniquePtr = std::unique_ptr<DType, CUDADeleter<DType>>;

protected:
  mutable cudaStream_t* streams { nullptr };
  mutable cublasHandle_t cublas_handle_default {};
  mutable cublasHandle_t* cublas_handle_array { nullptr };
  size_t num_allocated_streams { 0 };
  bool rng_initialized { false };

  mutable std::unique_ptr<curandState, models::common::cuda::CUDADeleter<curandState>> rng_state { nullptr };
  void setup_rng( unsigned long seed, const uint64_t size, const uint64_t batch_size );

  constexpr static float alpha = 1.0f;
  constexpr static float beta = 0.0f;

  constexpr static cublasComputeType_t GH_CUDA_COMPUTE_TYPE = CUBLAS_COMPUTE_32F;

public:
  Operations( const size_t num_streams,
              const size_t rng_states,
              const bool needs_rng = true,
              const bool needs_streams = true,
              const size_t batch_size = 1 );
  ~Operations();

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

  void soft_sample( DType* v, const std::vector<float>& tempratures, const size_t vocab_size ) const;

  DeviceUniquePtr device_allocate( const uint64_t size_bytes ) const;

  void randomize_device_buffer( DType* buffer, const uint64_t len, const float min, const float max ) const;

  void copy( DType* dst,
             const DType* l,
             const uint64_t batch_size,
             const CopyType type,
             const bool async = false ) const;

  void print( const DType* x, const uint64_t b, const std::string base ) const;
};

static_assert( OperationsConcept<Operations<orthrus::float32_t>, orthrus::float32_t> );
static_assert( OperationsConcept<Operations<orthrus::float16_t>, orthrus::float16_t> );
static_assert( OperationsConcept<Operations<orthrus::bfloat16_t>, orthrus::bfloat16_t> );

// helper functions are in this anonymous namespace
namespace {

template<std::unsigned_integral T>
consteval T div_ceil( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

template<std::unsigned_integral T>
T div_ceil_runtime( const T x, const T y )
{
  return x / y + ( x % y != 0 );
}

template<typename DType>
__global__ void accum_cuda( DType* a, const DType* b, const uint64_t size )
{
  const uint64_t i = blockIdx.x * TPB + threadIdx.x;
  if ( i < size ) {
    a[i] += b[i];
  }
}

namespace { // rmsnorm

template<uint64_t size, typename DType>
__global__ void normalize_and_scale(
  DType* output,
  const DType* x,
  const DType* weight,
  const orthrus::float32_t* ss,
  typename std::enable_if<!std::is_same_v<DType, orthrus::float32_t>>::type* = nullptr )
{
  const uint64_t gb_i = threadIdx.x + blockIdx.x * TPB + blockIdx.y * size;
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const orthrus::float32_t denom = sqrtf( ss[blockIdx.y] / size + 1e-5f );

    if constexpr ( std::is_same_v<DType, orthrus::float16_t> ) {
      output[gb_i] = weight[i] * __float2half( __half2float( x[gb_i] ) / denom );
    } else if constexpr ( std::is_same_v<DType, orthrus::bfloat16_t> ) {
      output[gb_i] = weight[i] * __float2bfloat16( __bfloat162float( x[gb_i] ) / denom );
    } else {
      STATIC_ASSERT_NO_MATCH();
    }
  }
}

template<uint64_t size, typename DType>
__global__ void normalize_and_scale(
  orthrus::float32_t* output,
  const orthrus::float32_t* x,
  const orthrus::float32_t* weight,
  const orthrus::float32_t* ss,
  typename std::enable_if<std::is_same_v<DType, orthrus::float32_t>>::type* = nullptr )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;

  if ( i < size ) {
    const orthrus::float32_t denom = sqrtf( *ss / size + 1e-5f );
    output[i] = weight[i] * x[i] / denom;
  }
}

template<typename DType>
__global__ void reduce_norm_v2_square_batched(
  orthrus::float32_t* output,
  const DType* x,
  const uint64_t size,
  typename std::enable_if<!std::is_same_v<DType, orthrus::float32_t>>::type* = nullptr )
{
  extern __shared__ orthrus::float32_t s_out[];

  const uint64_t global_tid = size * blockIdx.y + NRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = NRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                    // index within block

  if ( local_tid < size ) {
    const orthrus::float32_t _x_f = static_cast<orthrus::float32_t>( x[global_tid] );
    s_out[tid] = _x_f * _x_f;
  } else {
    s_out[tid] = 0;
  }
  if ( local_tid + NRBS < size ) {
    const orthrus::float32_t _x_f = static_cast<orthrus::float32_t>( x[global_tid + NRBS] );
    s_out[tid + NRBS] = _x_f * _x_f;
  } else {
    s_out[tid + NRBS] = 0;
  }

  for ( unsigned int s = NRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      s_out[tid] += s_out[tid + s];
    }
    __syncthreads();
  }

  if ( tid == 0 )
    output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0] + s_out[1];
}

__global__ void reduce_norm_v2_sum_batched( orthrus::float32_t* output,
                                            const orthrus::float32_t* x,
                                            const uint64_t size )
{
  extern __shared__ orthrus::float32_t s_out[];

  const uint64_t global_tid = size * blockIdx.y + NRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = NRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                    // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
  } else {
    s_out[tid] = 0;
  }
  if ( local_tid + NRBS < size ) {
    s_out[tid + NRBS] = x[global_tid + NRBS];
  } else {
    s_out[tid + NRBS] = 0;
  }

  for ( unsigned int s = NRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      s_out[tid] += s_out[tid + s];
    }
    __syncthreads();
  }

  if ( tid == 0 )
    output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0] + s_out[1];
}

template<uint64_t size>
void square_reduce_step_2( orthrus::float32_t* output,
                           const orthrus::float32_t* x,
                           orthrus::float32_t* temp_1,
                           orthrus::float32_t* temp_2,
                           const uint64_t batch_size )
{
  constexpr uint64_t max_elems_per_block = NRBS * 2;
  constexpr uint64_t shmem_size = sizeof( orthrus::float32_t ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if constexpr ( grid_size == 1 ) {
    reduce_norm_v2_sum_batched<<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    reduce_norm_v2_sum_batched<<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2<grid_size>( output, temp_1, temp_2, temp_1, batch_size );
  }
}

template<uint64_t size, typename DType>
void square_reduce_step_1( orthrus::float32_t* output,
                           const DType* x,
                           const uint64_t batch_size,
                           typename std::enable_if<!std::is_same_v<DType, orthrus::float32_t>>::type* = nullptr )
{
  constexpr uint64_t max_elems_per_block = NRBS * 2;
  constexpr uint64_t shmem_size = sizeof( orthrus::float32_t ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if constexpr ( grid_size == 1 ) {
    reduce_norm_v2_square_batched<<<grids, NRBS, shmem_size>>>( output, x, size );
  } else {
    orthrus::float32_t* temp_1 = output + batch_size;
    orthrus::float32_t* temp_2 = temp_1 + batch_size * grid_size;
    reduce_norm_v2_square_batched<<<grids, NRBS, shmem_size>>>( temp_1, x, size );
    square_reduce_step_2<grid_size>( output, temp_1, temp_2, temp_1, batch_size );
  }
}

} // namespace rmsnorm

namespace { // argmax

template<typename DType, uint64_t size>
__global__ void argmax_batched_init( uint32_t* output_arg, DType* output, const DType* x )
{
  extern __shared__ orthrus::float32_t smem[];

  DType* s_out = reinterpret_cast<DType*>( &smem[0] );
  uint32_t* a_out = reinterpret_cast<uint32_t*>( s_out + AMRBS * 2 );

  const uint64_t global_tid = size * blockIdx.y + AMRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = AMRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                     // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
    a_out[tid] = local_tid;
  } else {
    s_out[tid] = -get_dtype_infinity<DType>();
  }
  if ( local_tid + AMRBS < size ) {
    s_out[tid + AMRBS] = x[global_tid + AMRBS];
    a_out[tid + AMRBS] = local_tid + AMRBS;
  } else {
    s_out[tid + AMRBS] = -get_dtype_infinity<DType>();
  }

  for ( unsigned int s = AMRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      if ( s_out[tid + s] > s_out[tid] ) {
        s_out[tid] = s_out[tid + s];
        a_out[tid] = a_out[tid + s];
      }
    }
    __syncthreads();
  }

  if ( tid == 0 ) {
    if ( s_out[1] > s_out[0] ) {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[1];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[1];
    } else {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[0];
    }
  }
}

template<typename DType, uint64_t size>
__global__ void argmax_batched_next( uint32_t* output_arg, DType* output, const uint32_t* x_arg, const DType* x )
{
  extern __shared__ orthrus::float32_t smem[];

  DType* s_out = reinterpret_cast<DType*>( &smem[0] );
  uint32_t* a_out = reinterpret_cast<uint32_t*>( s_out + AMRBS * 2 );

  const uint64_t global_tid = size * blockIdx.y + AMRBS * 2 * blockIdx.x + threadIdx.x; // index within whole batch
  const uint64_t local_tid = AMRBS * 2 * blockIdx.x + threadIdx.x;                      // index within array
  const uint64_t tid = threadIdx.x;                                                     // index within block

  if ( local_tid < size ) {
    s_out[tid] = x[global_tid];
    a_out[tid] = x_arg[global_tid];
  } else {
    s_out[tid] = -get_dtype_infinity<DType>();
  }
  if ( local_tid + AMRBS < size ) {
    s_out[tid + AMRBS] = x[global_tid + AMRBS];
    a_out[tid + AMRBS] = x_arg[global_tid + AMRBS];
  } else {
    s_out[tid + AMRBS] = -get_dtype_infinity<DType>();
  }

  for ( unsigned int s = AMRBS; s > 1; s >>= 1 ) {
    if ( tid < s ) {
      if ( s_out[tid + s] > s_out[tid] ) {
        s_out[tid] = s_out[tid + s];
        a_out[tid] = a_out[tid + s];
      }
    }
    __syncthreads();
  }

  if ( tid == 0 ) {
    if ( s_out[1] > s_out[0] ) {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[1];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[1];
    } else {
      output[blockIdx.y * gridDim.x + blockIdx.x] = s_out[0];
      output_arg[blockIdx.y * gridDim.x + blockIdx.x] = a_out[0];
    }
  }
}

template<typename DType, uint64_t size>
void argmax_step_2( uint32_t* output_arg,
                    const uint32_t* x_arg,
                    const DType* x,
                    uint32_t* temp_1_arg,
                    DType* temp_1,
                    uint32_t* temp_2_arg,
                    DType* temp_2,
                    const uint64_t batch_size )
{
  constexpr uint64_t max_elems_per_block = AMRBS * 2;
  constexpr uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );

  if constexpr ( grid_size == 1 ) {
    argmax_batched_next<DType, size><<<grids, AMRBS, shmem_size>>>( output_arg, temp_1, x_arg, x );
  } else {
    argmax_batched_next<DType, size><<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x_arg, x );
    argmax_step_2<DType, grid_size>(
      output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, batch_size );
  }
}

template<typename DType, uint64_t size>
void argmax_step_1( uint32_t* output_arg, const DType* x, const uint64_t batch_size )
{
  constexpr uint64_t max_elems_per_block = AMRBS * 2;
  constexpr uint64_t shmem_size = ( sizeof( DType ) + sizeof( uint32_t ) ) * max_elems_per_block;
  constexpr uint64_t grid_size = div_ceil( size, max_elems_per_block );

  dim3 grids( grid_size, batch_size );
  if constexpr ( grid_size == 1 ) {
    DType* output = reinterpret_cast<DType*>( output_arg + batch_size );
    argmax_batched_init<DType, size><<<grids, AMRBS, shmem_size>>>( output_arg, output, x );
  } else {
    DType* temp_1 = reinterpret_cast<DType*>( output_arg + batch_size );
    DType* temp_2 = temp_1 + batch_size * grid_size;
    uint32_t* temp_1_arg = reinterpret_cast<uint32_t*>( temp_2 + batch_size * grid_size );
    uint32_t* temp_2_arg = temp_1_arg + batch_size * grid_size;
    argmax_batched_init<DType, size><<<grids, AMRBS, shmem_size>>>( temp_1_arg, temp_1, x );
    argmax_step_2<DType, grid_size>(
      output_arg, temp_1_arg, temp_1, temp_2_arg, temp_2, temp_1_arg, temp_1, batch_size );
  }
}

} // namespace argmax

namespace { // silu

template<typename DType>
__global__ void silu_direct( DType* _hb, const DType* _hb2, const uint64_t hidden_dim )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;
  if ( i < hidden_dim ) {
    const DType x = _hb[i];
    _hb[i] = x / ( DType( 1.0f ) + hexp( -x ) ) * _hb2[i];
  }
}

template<>
__global__ void silu_direct( orthrus::float32_t* _hb, const orthrus::float32_t* _hb2, const uint64_t hidden_dim )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;
  if ( i < hidden_dim ) {
    const orthrus::float32_t x = _hb[i];
    _hb[i] = x / ( 1.0f + expf( -x ) ) * _hb2[i];
  }
}

}

namespace { // soft_sample

template<typename DType, size_t LPK>
__global__ void gumbel_fix( DType* array,
                            const orthrus::float32_t temp,
                            const size_t vocab_size,
                            curandState* rng_state )
{
  const uint64_t i = threadIdx.x + blockIdx.x * TPB;
  curandState local_state = rng_state[i];

  const size_t j_start = i * LPK;
  const size_t j_end = ( i + 1 ) * LPK > vocab_size ? vocab_size : ( i + 1 ) * LPK;

  for ( size_t j = j_start; j < j_end; j++ ) {
    orthrus::float32_t myrandf = curand_uniform( &local_state );
    myrandf = logf( -logf( myrandf ) );

    if constexpr ( std::is_same_v<DType, orthrus::float16_t> ) {
      array[j] = __float2half( __half2float( array[j] ) / temp - myrandf );
    } else if constexpr ( std::is_same_v<DType, orthrus::bfloat16_t> ) {
      array[j] = __float2bfloat16( __bfloat162float( array[j] ) / temp - myrandf );
    } else if constexpr ( std::is_same_v<DType, orthrus::float32_t> ) {
      array[j] = array[j] / temp - myrandf;
    } else {
      STATIC_ASSERT_NO_MATCH();
    }
  }

  rng_state[i] = local_state;
}

}

namespace { // setup_rng

__global__ void setup_rng_kernel( curandState* state, unsigned long seed )
{
  int id = threadIdx.x + blockIdx.x * TPB;
  curand_init( seed, id, 0, &state[id] );
}

}

namespace { // print

template<typename DType>
__global__ void print_cuda( const DType* x, const uint64_t b )
{
  for ( uint64_t i = 0; i < b; i++ ) {
    orthrus::float32_t c = static_cast<orthrus::float32_t>( x[i] );
    printf( "\t%.10f", c );
  }
  printf( "\n" );
}

}

} // end of anonymous namespace for helper functions

template<typename DType>
Operations<DType>::Operations( const size_t num_streams,
                               const size_t rng_states,
                               const bool needs_rng,
                               const bool needs_streams,
                               const size_t batch_size )
{
  // setup cuBLAS
  CHECK_CUBLAS( cublasCreate( &cublas_handle_default ) );

  if ( needs_streams ) {
    num_allocated_streams = num_streams > MAX_STREAMS ? MAX_STREAMS : num_streams;
    streams = (cudaStream_t*)malloc( num_allocated_streams * sizeof( cudaStream_t ) );
    cublas_handle_array = (cublasHandle_t*)malloc( num_allocated_streams * sizeof( cublasHandle_t ) );
    for ( size_t i = 0; i < num_allocated_streams; i++ ) {
      CHECK_CUDA( cudaStreamCreate( &( streams[i] ) ) );
      CHECK_CUBLAS( cublasCreate( &( cublas_handle_array[i] ) ) );
      CHECK_CUBLAS( cublasSetStream( cublas_handle_array[i], streams[i] ) );
    }
  }

  if ( needs_rng ) {
    // setup cuRAND; must be done after setting up the streams
    setup_rng( 1234u, rng_states, batch_size );
  }
}

template<typename DType>
Operations<DType>::~Operations()
{
  CHECK_CUBLAS( cublasDestroy( cublas_handle_default ) );
  for ( size_t i = 0; i < num_allocated_streams; i++ ) {
    CHECK_CUBLAS( cublasDestroy( cublas_handle_array[i] ) );
    cudaStreamDestroy( streams[i] );
  }
  free( streams );
  free( cublas_handle_array );
}

template<typename DType>
void Operations<DType>::print( const DType* x, const uint64_t b, const std::string base ) const
{
  printf( "%s", base.c_str() );
  cudaDeviceSynchronize();
  print_cuda<<<1, 1>>>( x, b );
  cudaDeviceSynchronize();
}

template<typename DType>
template<uint64_t size>
void Operations<DType>::accum( DType* a, const DType* b, const uint64_t batch_size ) const
{
  accum_cuda<<<div_ceil_runtime( size * batch_size, TPB ), TPB>>>( a, b, size * batch_size );
}

template<>
template<uint64_t size>
void Operations<orthrus::float32_t>::accum( orthrus::float32_t* a,
                                              const orthrus::float32_t* b,
                                              const uint64_t batch_size ) const
{
  CHECK_CUBLAS( cublasSaxpy( cublas_handle_default, size * batch_size, &alpha, b, 1, a, 1 ) );
}

template<typename DType>
template<uint64_t size>
void Operations<DType>::rmsnorm( DType* output,
                                 const DType* x,
                                 orthrus::float32_t* temp,
                                 const DType* weight,
                                 const uint64_t batch_size ) const
{
  dim3 grid { div_ceil( size, TPB ), static_cast<uint32_t>( batch_size ) };
  square_reduce_step_1<size>( temp, x, batch_size );
  normalize_and_scale<size, DType><<<grid, TPB>>>( output, x, weight, temp );
}

template<>
template<uint64_t size>
void Operations<orthrus::float32_t>::rmsnorm( orthrus::float32_t* output,
                                                const orthrus::float32_t* x,
                                                orthrus::float32_t* temp,
                                                const orthrus::float32_t* weight,
                                                const uint64_t batch_size ) const
{
  for ( size_t i = 0; i < batch_size; i++ ) {
    CHECK_CUBLAS( cublasSdot( cublas_handle_default, size, x + i * size, 1, x + i * size, 1, temp + i ) );
    normalize_and_scale<size, DType>
      <<<div_ceil( size, TPB ), TPB>>>( output + i * size, x + i * size, weight, temp + i );
  }
}

template<typename DType>
template<uint64_t n>
void Operations<DType>::argmax( uint32_t* output, const DType* v, DType* temp, const uint64_t batch_size ) const
{
  argmax_step_1<DType, n>( reinterpret_cast<uint32_t*>( temp ), v, batch_size );
  CHECK_CUDA( cudaMemcpy( output, temp, batch_size * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
}

template<typename DType>
template<uint64_t hidden_dim>
void Operations<DType>::silu( DType* hb, DType* hb2, const uint64_t batch_size ) const
{
  silu_direct<<<div_ceil_runtime( hidden_dim * batch_size, TPB ), TPB>>>( hb, hb2, hidden_dim * batch_size );
}

template<typename DType>
template<uint64_t s, uint64_t r>
void Operations<DType>::matmul( DType* xout, const DType* x, const DType* W, const uint64_t b ) const
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)
  const uint64_t m = r;
  const uint64_t n = b;
  const uint64_t k = s;
  const uint64_t lda = k;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  CHECK_CUBLAS( cublasGemmEx( cublas_handle_default,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              W,
                              get_cuda_data_type<DType>(),
                              lda,
                              x,
                              get_cuda_data_type<DType>(),
                              ldb,
                              &beta,
                              xout,
                              get_cuda_data_type<DType>(),
                              ldc,
                              GH_CUDA_COMPUTE_TYPE,
                              CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
void Operations<DType>::untemplated_matmul( DType* xout,
                                            const DType* x,
                                            const DType* W,
                                            const uint64_t s,
                                            const uint64_t r,
                                            const uint64_t b ) const
{
  // x(b,s) @ W(s,r) -> xout(b,r)
  // OR
  // W(r,s) @ x(s,b) -> xout(r,b)
  // A(m,k) @ B(k,n) ->    C(m,n)
  const uint64_t m = r;
  const uint64_t n = b;
  const uint64_t k = s;
  const uint64_t lda = k;
  const uint64_t ldb = k;
  const uint64_t ldc = m;

  CHECK_CUBLAS( cublasGemmEx( cublas_handle_default,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              W,
                              get_cuda_data_type<DType>(),
                              lda,
                              x,
                              get_cuda_data_type<DType>(),
                              ldb,
                              &beta,
                              xout,
                              get_cuda_data_type<DType>(),
                              ldc,
                              GH_CUDA_COMPUTE_TYPE,
                              CUBLAS_GEMM_DEFAULT ) );
}

template<typename DType>
void Operations<DType>::soft_sample( DType* v, const std::vector<float>& temperatures, const size_t vocab_size ) const
{
  DCHECK_GT( num_allocated_streams, 0 ) << "This operation currently needs streams, but they are not allocated";
  const size_t rng_size = div_ceil_runtime( vocab_size, LPK );
  for ( uint64_t i = 0; i < temperatures.size(); i++ ) {
    if ( temperatures[i] > 0 ) {
      gumbel_fix<DType, LPK><<<div_ceil_runtime( rng_size, TPB ), TPB, 0, this->streams[i % MAX_STREAMS]>>>(
        v + i * vocab_size, temperatures[i], vocab_size, rng_state.get() + i * rng_size );
    }
  }
}

template<typename DType>
Operations<DType>::DeviceUniquePtr Operations<DType>::device_allocate( const uint64_t size_bytes ) const
{
  LOG( WARNING ) << "Allocating pointer with size " << ( size_bytes >> 20 ) << " MiB...";
  DType* ptr;
  CHECK_CUDA( cudaMalloc( &ptr, size_bytes ) );
  LOG( WARNING ) << "Allocated...";
  return DeviceUniquePtr { ptr };
}

template<typename DType>
void Operations<DType>::copy( DType* dst,
                              const DType* l,
                              const uint64_t size_bytes,
                              const CopyType type,
                              const bool async ) const
{
  auto convert_to_cuda = []( const CopyType copy_type ) {
    switch ( copy_type ) {
      case CopyType::HostToHost: return cudaMemcpyHostToHost;
      case CopyType::HostToDevice: return cudaMemcpyHostToDevice;
      case CopyType::DeviceToHost: return cudaMemcpyDeviceToHost;
      case CopyType::DeviceToDevice: return cudaMemcpyDeviceToDevice;
      default: return cudaMemcpyDefault;
    }
  };

  if ( async ) {
    CHECK_CUDA( cudaMemcpyAsync( dst, l, size_bytes, convert_to_cuda( type ) ) );
  } else {
    CHECK_CUDA( cudaMemcpy( dst, l, size_bytes, convert_to_cuda( type ) ) );
  }
}

template<typename DType>
void Operations<DType>::randomize_device_buffer( DType* buffer,
                                                 const uint64_t len,
                                                 const float min,
                                                 const float max ) const
{
  std::unique_ptr<DType[]> host_buffer { new DType[len] };
  util::randomize_buffer( host_buffer.get(), len, min, max );
  CHECK_CUDA( cudaMemcpy( buffer, host_buffer.get(), len * sizeof( DType ), cudaMemcpyHostToDevice ) );
}

template<typename DType>
void Operations<DType>::setup_rng( unsigned long seed, const uint64_t size, const uint64_t batch_size )
{
  curandState* rng_state_ptr = nullptr;
  const size_t rng_size = div_ceil_runtime( size, LPK );
  LOG( WARNING ) << "Allocating memory for RNG state (" << ( rng_size * batch_size * sizeof( curandState ) >> 20 )
                 << " MiB)";
  common::cuda::CHECK_CUDA( cudaMalloc( &rng_state_ptr, rng_size * batch_size * sizeof( curandState ) ) );
  rng_state.reset( rng_state_ptr );

  for ( uint64_t i = 0; i < batch_size; i++ ) {
    setup_rng_kernel<<<div_ceil_runtime( rng_size, TPB ), TPB, 0>>>( rng_state.get() + i * rng_size, seed );
  }
  rng_initialized = true;
}

} // namespace orthrus::models::common::cuda
