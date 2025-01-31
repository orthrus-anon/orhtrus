#pragma once

#if defined( TARGET_PLATFORM_AMD64 )
#define _ORTHRUS_ARCH_NS_ amd64
#define _ORTHRUS_PLATFORM_NAME_ AMD64
#endif

#if defined( TARGET_PLATFORM_CUDA )
#define _ORTHRUS_ARCH_NS_ cuda
#define _ORTHRUS_PLATFORM_NAME_ CUDA
#endif

#if defined( TARGET_DTYPE_FLOAT16 )
#define _ORTHRUS_DTYPE_NAME_ Float16
#if defined( TARGET_PLATFORM_AMD64 )
#define _ORTHRUS_DTYPE_ _Float16
#elif defined( TARGET_PLATFORM_CUDA )
#define _ORTHRUS_DTYPE_ __half
#endif
#endif

#if defined( TARGET_DTYPE_FLOAT32 )
#define _ORTHRUS_DTYPE_NAME_ Float32
#define _ORTHRUS_DTYPE_ float
#endif

#if defined( TARGET_DTYPE_BFLOAT16 )
#define _ORTHRUS_DTYPE_NAME_ BFloat16
#if defined( TARGET_PLATFORM_CUDA )
#define _ORTHRUS_DTYPE_ __nv_bfloat16
#elif defined( TARGET_PLATFORM_AMD64 )
#error "BFloat16 is not supported on AMD64"
#endif
#endif

#if !( defined( _ORTHRUS_ARCH_NS_ ) && defined( _ORTHRUS_PLATFORM_NAME_ ) && defined( _ORTHRUS_DTYPE_ ) )
#error "TARGET_PLATFORM and TARGET_DTYPE must be defined"
#endif
