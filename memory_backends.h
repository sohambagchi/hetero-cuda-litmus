// memory_backends.h — Memory allocation abstraction for 3 het backends
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#ifdef HET_DEBUG
#define HET_MEMORY_DEBUG_LOG(...) do { \
  fprintf(stderr, "[HET_DEBUG][memory] " __VA_ARGS__); \
  fprintf(stderr, "\n"); \
} while(0)
#else
#define HET_MEMORY_DEBUG_LOG(...) do { } while(0)
#endif

#define HET_CHECK(err) do { \
  cudaError_t e = (err); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

// Allocate memory accessible by both CPU and GPU.
// Sets both *host_ptr and *device_ptr.
inline void het_malloc(void** host_ptr, void** device_ptr, size_t size) {
  HET_MEMORY_DEBUG_LOG("het_malloc request size=%zu bytes", size);
#if defined(MEM_HOSTALLOC)
  HET_CHECK(cudaHostAlloc(host_ptr, size, cudaHostAllocMapped));
  HET_CHECK(cudaHostGetDevicePointer(device_ptr, *host_ptr, 0));
#elif defined(MEM_MANAGED)
  HET_CHECK(cudaMallocManaged(host_ptr, size));
  *device_ptr = *host_ptr;
#elif defined(MEM_MALLOC)
  // Plain malloc — unified address space on SoC (e.g., Jetson, Grace-Hopper)
  *host_ptr = malloc(size);
  if (*host_ptr == nullptr) {
    fprintf(stderr, "malloc failed: %zu bytes\n", size);
    exit(EXIT_FAILURE);
  }
  *device_ptr = *host_ptr;
#else
  #error "Must define MEM_HOSTALLOC, MEM_MANAGED, or MEM_MALLOC"
#endif

  HET_MEMORY_DEBUG_LOG("het_malloc complete host_ptr=%p device_ptr=%p size=%zu bytes", *host_ptr, *device_ptr, size);
}

// Free memory allocated by het_malloc.
inline void het_free(void* host_ptr) {
  HET_MEMORY_DEBUG_LOG("het_free host_ptr=%p", host_ptr);
#if defined(MEM_HOSTALLOC)
  HET_CHECK(cudaFreeHost(host_ptr));
#elif defined(MEM_MANAGED)
  HET_CHECK(cudaFree(host_ptr));
#elif defined(MEM_MALLOC)
  free(host_ptr);
#endif
}

// Memset that works for all backends (host-side).
inline void het_memset(void* ptr, int value, size_t size) {
  HET_MEMORY_DEBUG_LOG("het_memset ptr=%p value=%d size=%zu bytes", ptr, value, size);
  memset(ptr, value, size);
#if defined(MEM_MANAGED)
  // Ensure GPU sees the memset before kernel launch
  HET_MEMORY_DEBUG_LOG("het_memset forcing cudaDeviceSynchronize for managed memory");
  cudaDeviceSynchronize();
#endif
}
