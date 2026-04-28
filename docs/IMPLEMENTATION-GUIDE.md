# IMPLEMENTATION-GUIDE.md — Per-File Specifications

This document provides exact specifications for every file in the framework: data structures, function signatures, code patterns, and implementation details. A developer should be able to implement each file from this spec alone.

---

## Table of Contents

1. [litmus_het.cuh — Core Header](#1-litmus_hetchuh--core-header)
2. [memory_backends.h — Memory Allocation Abstraction](#2-memory_backendsh--memory-allocation-abstraction)
3. [functions.cu — GPU Device Functions](#3-functionscu--gpu-device-functions)
4. [cpu_functions.h — CPU-Side Functions](#4-cpu_functionsh--cpu-side-functions)
5. [runner.cu — Host-Side Runner](#5-runnercu--host-side-runner)
6. [Kernel File Pattern](#6-kernel-file-pattern)
7. [tune.sh — Tuning Script](#7-tunesh--tuning-script)
8. [analyze.py — Analysis Script](#8-analyzepy--analysis-script)
9. [Parameter Files](#9-parameter-files)
10. [Tuning Files](#10-tuning-files)

---

## 1. litmus_het.cuh — Core Header

### Purpose
Central header included by all kernel files and runner.cu. Defines atomic types, macros, data structures, and function declarations.

### Atomic Type Definitions

```cpp
#include <cuda_runtime.h>
#include <cuda/atomic>

// Shared backing storage. GPU operations use TEST_SCOPE through atomic_ref,
// while CPU operations stay system-scope.
struct het_atomic_uint {
  uint value;

  template <cuda::thread_scope Scope>
  __host__ __device__ cuda::atomic_ref<uint, Scope> ref() {
    return cuda::atomic_ref<uint, Scope>(value);
  }

  __host__ __device__ void store(uint desired,
                                 cuda::memory_order order = cuda::memory_order_seq_cst) {
#ifdef __CUDA_ARCH__
    ref<TEST_SCOPE>().store(desired, order);
#else
    ref<cuda::thread_scope_system>().store(desired, order);
#endif
  }

  __host__ __device__ uint load(
      cuda::memory_order order = cuda::memory_order_seq_cst) const {
#ifdef __CUDA_ARCH__
    return cuda::atomic_ref<const uint, TEST_SCOPE>(value).load(order);
#else
    return cuda::atomic_ref<const uint, cuda::thread_scope_system>(value).load(order);
#endif
  }
};

// Operation scope — controls memory_order scope for test operations
#ifdef SCOPE_DEVICE
  #define TEST_SCOPE cuda::thread_scope_device
#elif defined(SCOPE_BLOCK)
  #define TEST_SCOPE cuda::thread_scope_block
#elif defined(SCOPE_SYSTEM)
  #define TEST_SCOPE cuda::thread_scope_system
#else
  #define TEST_SCOPE cuda::thread_scope_system  // default for het
#endif

// Fence scope — controls scope for explicit fences
#ifdef FENCE_SCOPE_BLOCK
  #define FENCE_SCOPE cuda::thread_scope_block
#elif defined(FENCE_SCOPE_DEVICE)
  #define FENCE_SCOPE cuda::thread_scope_device
#elif defined(FENCE_SCOPE_SYSTEM)
  #define FENCE_SCOPE cuda::thread_scope_system
#else
  #define FENCE_SCOPE cuda::thread_scope_system  // default for het
#endif
```

### HET Split Macros

These macros define which thread roles execute on CPU vs GPU for each test instance. Each HET_ define implies a set of `THREAD_N_IS_CPU` / `THREAD_N_IS_GPU` defines.

```cpp
// 2-thread splits
#ifdef HET_C0_G1
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 1
#elif defined(HET_C1_G0)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 1
#endif

// 3-thread splits (all combinations with at least 1 CPU and 1 GPU)
#ifdef HET_C0_G1_G2
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 2
#elif defined(HET_C0_C1_G2)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 1
// ... (all 6 valid 3-thread splits: C0_G1_G2, C1_G0_G2, C2_G0_G1, C0_C1_G2, C0_C2_G1, C1_C2_G0)
#endif

// 4-thread splits (all combinations with at least 1 CPU and 1 GPU)
// 14 valid splits: 4C1×GPU_rest + 4C2×GPU_rest + 4C3×GPU_rest = 4 + 6 + 4 = 14
// Examples: HET_C0_G1_G2_G3, HET_C0_C1_G2_G3, HET_C0_C1_C2_G3, etc.
```

### TB_ Macros — GPU Thread ID Computation

These are identical to cuda-litmus, but only apply to GPU-side threads. The `DEFINE_IDS()` macro computes `id_0, id_1, ...` which are used by GPU threads to determine which test instance they handle.

**Keep the full set from cuda-litmus** (TB_0_1, TB_01, TB_0_1_2, TB_01_2, TB_0_12, TB_02_1, TB_012, and all 4-thread variants).

```cpp
// Exactly as in cuda-litmus litmus.cuh lines 28-300
// No changes needed — these only run on GPU
```

### Memory Location Macros

Same as cuda-litmus, but operating on `het_atomic_uint*` shared storage:

```cpp
#define TWO_THREAD_TWO_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2; \
  uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) \
             * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2; \
  uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) \
             * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

// THREE_THREAD_TWO_MEM_LOCATIONS(), THREE_THREAD_THREE_MEM_LOCATIONS()
// same as cuda-litmus
```

### Stress Macros (GPU side)

```cpp
#define PRE_STRESS() \
  if (kernel_params->pre_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, \
              kernel_params->pre_stress_pattern); \
  } \
  if (kernel_params->barrier) { \
    spin(gpu_barrier, blockDim.x * kernel_params->testing_workgroups); \
  }

#define MEM_STRESS() \
  else if (kernel_params->mem_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, \
              kernel_params->mem_stress_iterations); \
  }
```

### Data Structures

```cpp
typedef struct {
  uint r0;
  uint r1;
  uint r2;
  uint r3;
  uint r4;
  uint r5;
  uint r6;
  uint r7;
} ReadResults;

typedef struct {
  uint t0;
  uint t1;
  uint t2;
  uint t3;
  uint x;
  uint y;
  uint z;
} TestInstance;

typedef struct {
  cuda::atomic<uint, cuda::thread_scope_system> res0;
  cuda::atomic<uint, cuda::thread_scope_system> res1;
  cuda::atomic<uint, cuda::thread_scope_system> res2;
  cuda::atomic<uint, cuda::thread_scope_system> res3;
  cuda::atomic<uint, cuda::thread_scope_system> res4;
  cuda::atomic<uint, cuda::thread_scope_system> res5;
  cuda::atomic<uint, cuda::thread_scope_system> res6;
  cuda::atomic<uint, cuda::thread_scope_system> res7;
  cuda::atomic<uint, cuda::thread_scope_system> res8;
  cuda::atomic<uint, cuda::thread_scope_system> res9;
  cuda::atomic<uint, cuda::thread_scope_system> res10;
  cuda::atomic<uint, cuda::thread_scope_system> res11;
  cuda::atomic<uint, cuda::thread_scope_system> res12;
  cuda::atomic<uint, cuda::thread_scope_system> res13;
  cuda::atomic<uint, cuda::thread_scope_system> res14;
  cuda::atomic<uint, cuda::thread_scope_system> res15;
  cuda::atomic<uint, cuda::thread_scope_system> weak;
  cuda::atomic<uint, cuda::thread_scope_system> na;
  cuda::atomic<uint, cuda::thread_scope_system> other;
} TestResults;
// NOTE: system scope because check_results may run on either CPU or GPU

typedef struct {
  // GPU stress params (from cuda-litmus)
  bool barrier;
  bool mem_stress;
  int mem_stress_iterations;
  int mem_stress_pattern;
  bool pre_stress;
  int pre_stress_iterations;
  int pre_stress_pattern;
  int permute_thread;
  int permute_location;
  int testing_workgroups;
  int mem_stride;
  int mem_offset;
  // Het-specific params
  int total_instances;       // = workgroupSize * testing_workgroups
  int barrier_spin_limit;    // max spin iterations for per-instance barrier
} KernelParams;

// Per-instance barrier for CPU-GPU synchronization
// Each logical instance expects exactly one CPU arrival and one GPU arrival.
#define HET_BARRIER_EXPECTED 2
typedef cuda::atomic<uint, cuda::thread_scope_system> het_barrier_t;
```

### Function Declarations

```cpp
// GPU kernel — litmus test (runs GPU-side thread operations)
__global__ void litmus_test(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* gpu_barrier,  // intra-GPU barrier
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances,
  het_barrier_t* het_barriers);  // NEW: per-instance CPU-GPU barriers

// GPU kernel — result classification
__global__ void check_results(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak);

// Host function — print results, return weak count
int host_check_results(TestResults* results, bool print);

// Host function — CPU thread entry point for test execution
void cpu_test_thread(
  int thread_id,
  int start_instance,
  int end_instance,
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  KernelParams* kernel_params,
  het_barrier_t* het_barriers,
  uint* cpu_scratchpad,
  bool cpu_pre_stress,
  int cpu_pre_stress_iterations,
  int cpu_pre_stress_pattern);
```

---

## 2. memory_backends.h — Memory Allocation Abstraction

### Purpose
Provides a unified interface for allocating/freeing shared CPU-GPU memory across three backends.

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

// Returns a pointer usable by both CPU and GPU
// Sets both *host_ptr and *device_ptr
inline void het_malloc(void** host_ptr, void** device_ptr, size_t size) {
#if defined(MEM_HOSTALLOC)
  cudaHostAlloc(host_ptr, size, cudaHostAllocMapped);
  cudaHostGetDevicePointer(device_ptr, *host_ptr, 0);
#elif defined(MEM_MANAGED)
  cudaMallocManaged(host_ptr, size);
  *device_ptr = *host_ptr;
#elif defined(MEM_MALLOC)
  *host_ptr = malloc(size);
  *device_ptr = *host_ptr;  // unified address space on SoC
#else
  #error "Must define MEM_HOSTALLOC, MEM_MANAGED, or MEM_MALLOC"
#endif
}

inline void het_free(void* host_ptr) {
#if defined(MEM_HOSTALLOC)
  cudaFreeHost(host_ptr);
#elif defined(MEM_MANAGED)
  cudaFree(host_ptr);
#elif defined(MEM_MALLOC)
  free(host_ptr);
#endif
}

// Memset that works for all backends (host-side)
inline void het_memset(void* ptr, int value, size_t size) {
  memset(ptr, value, size);
#if defined(MEM_MANAGED)
  // Ensure GPU sees the memset before kernel launch
  cudaDeviceSynchronize();
#endif
}
```

---

## 3. functions.cu — GPU Device Functions

### Purpose
Identical to cuda-litmus's `functions.cu`. No changes needed.

```cpp
__device__ uint permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

__device__ uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1) % testing_workgroups;
}

__device__ void spin(cuda::atomic<uint, cuda::thread_scope_device>* barrier, uint limit) {
  int i = 0;
  uint val = barrier->fetch_add(1, cuda::memory_order_relaxed);
  while (i < 1024 && val < limit) {
    val = barrier->load(cuda::memory_order_relaxed);
    i++;
  }
}

__device__ void do_stress(uint* scratchpad, uint* scratch_locations, uint iterations, uint pattern) {
  // Identical to cuda-litmus (4 patterns: SS, SL, LS, LL)
  for (uint i = 0; i < iterations; i++) {
    if (pattern == 0) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      scratchpad[scratch_locations[blockIdx.x]] = i + 1;
    } else if (pattern == 1) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) break;
    } else if (pattern == 2) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) break;
      scratchpad[scratch_locations[blockIdx.x]] = i;
    } else if (pattern == 3) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) break;
      uint tmp2 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp2 > 100) break;
    }
  }
}
```

### NEW: Het barrier spin (GPU side)

```cpp
// GPU thread waits at per-instance het barrier
__device__ void het_spin(het_barrier_t* barrier, uint expected_count, uint spin_limit) {
  barrier->fetch_add(1, cuda::memory_order_relaxed);
  for (uint i = 0; i < spin_limit; i++) {
    if (barrier->load(cuda::memory_order_acquire) >= expected_count) return;
  }
}
```

---

## 4. cpu_functions.h — CPU-Side Functions

### Purpose
CPU-side equivalents of GPU stress and barrier functions.

```cpp
#pragma once
#include <cuda/atomic>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>

// CPU-side permute_id (same formula as GPU)
inline uint cpu_permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

// CPU-side het barrier spin
inline void cpu_het_spin(het_barrier_t* barrier, uint expected_count, uint spin_limit) {
  barrier->fetch_add(1, cuda::memory_order_relaxed);
  for (uint i = 0; i < spin_limit; i++) {
    if (barrier->load(cuda::memory_order_acquire) >= expected_count) return;
  }
}

// CPU stress patterns (mirrors GPU do_stress but uses volatile uint* on CPU side)
inline void cpu_do_stress(volatile uint* scratchpad, uint iterations, uint pattern) {
  for (uint i = 0; i < iterations; i++) {
    if (pattern == 0) {
      scratchpad[0] = i;
      scratchpad[0] = i + 1;
    } else if (pattern == 1) {
      scratchpad[0] = i;
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
    } else if (pattern == 2) {
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
      scratchpad[0] = i;
    } else if (pattern == 3) {
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
      uint tmp2 = scratchpad[0];
      if (tmp2 > 100) break;
    }
  }
}

// Background CPU stress thread function
inline void cpu_memory_stress_thread(volatile uint* stress_array, int array_size,
                                     volatile bool* stop_flag) {
  while (!*stop_flag) {
    for (int i = 1; i < array_size - 1; i++) {
      stress_array[i] = 1 + stress_array[i - 1] + stress_array[i + 1];
    }
  }
}
```

---

## 5. runner.cu — Host-Side Runner

### Purpose
Host-side orchestrator: parses parameters, allocates memory, launches GPU kernel + CPU threads, collects results.

### Key Data Structures (host-side)

```cpp
typedef struct {
  int numMemLocations;
  int permuteLocation;
} TestParams;

typedef struct {
  // All 18 cuda-litmus params
  int testIterations;
  int testingWorkgroups;
  int maxWorkgroups;
  int workgroupSize;
  int shufflePct;
  int barrierPct;
  int stressLineSize;
  int stressTargetLines;
  int scratchMemorySize;
  int memStride;
  int memStressPct;
  int memStressIterations;
  int memStressPattern;
  int preStressPct;
  int preStressIterations;
  int preStressPattern;
  int stressAssignmentStrategy;
  int permuteThread;
  // Het-specific params
  int cpuStressThreads;         // 0 = auto (hardware_concurrency / 2)
  int cpuPreStressPct;          // 0-100
  int cpuPreStressIterations;   // 0-128
  int cpuPreStressPattern;      // 0-3
  int barrierSpinLimit;         // default 4096
} StressParams;
```

### Execution Flow (run function)

```cpp
void run(StressParams stressParams, TestParams testParams, bool print_results) {
  int testingThreads = stressParams.workgroupSize * stressParams.testingWorkgroups;
  int numCpuThreads = std::thread::hardware_concurrency();
  int instancesPerCpuThread = (testingThreads + numCpuThreads - 1) / numCpuThreads;

  // 1. Allocate shared memory (via het_malloc for all shared buffers)
  het_atomic_uint *h_testLocations, *d_testLocations;
  het_malloc((void**)&h_testLocations, (void**)&d_testLocations,
             testLocSize);

  ReadResults *h_readResults, *d_readResults;
  het_malloc((void**)&h_readResults, (void**)&d_readResults,
             sizeof(ReadResults) * testingThreads);

  het_barrier_t *h_hetBarriers, *d_hetBarriers;
  het_malloc((void**)&h_hetBarriers, (void**)&d_hetBarriers,
             sizeof(het_barrier_t) * testingThreads);

  TestResults *h_testResults, *d_testResults;
  het_malloc((void**)&h_testResults, (void**)&d_testResults,
             sizeof(TestResults));

  // GPU-only allocations (scratchpad, shuffled_workgroups, etc.) use cudaMalloc
  // because they don't need CPU access
  uint* d_shuffledWorkgroups;
  cudaMalloc(&d_shuffledWorkgroups, stressParams.maxWorkgroups * sizeof(uint));
  // ... etc for scratchpad, scratch_locations, gpu_barrier, kernel_params

  // CPU stress scratchpad (separate from GPU scratchpad)
  volatile uint* cpuScratchpad = (volatile uint*)malloc(4096 * sizeof(uint));

  // 2. Start background CPU stress threads
  volatile bool stopStress = false;
  std::vector<std::thread> stressThreads;
  int numStressThreads = stressParams.cpuStressThreads > 0
    ? stressParams.cpuStressThreads : numCpuThreads / 2;
  volatile uint* stressArray = (volatile uint*)malloc(1024 * sizeof(uint));
  for (int i = 0; i < numStressThreads; i++) {
    stressThreads.emplace_back(cpu_memory_stress_thread, stressArray, 1024, &stopStress);
  }

  // 3. Main iteration loop
  for (int iter = 0; iter < stressParams.testIterations; iter++) {
    // Clear shared memory
    het_memset(h_testLocations, 0, testLocSize);
    het_memset(h_readResults, 0, sizeof(ReadResults) * testingThreads);
    het_memset(h_hetBarriers, 0, sizeof(het_barrier_t) * testingThreads);
    het_memset(h_testResults, 0, sizeof(TestResults));

    // Randomize GPU params
    int numWorkgroups = setBetween(stressParams.testingWorkgroups, stressParams.maxWorkgroups);
    setShuffledWorkgroups(...);
    setScratchLocations(...);
    setDynamicKernelParams(...);
    // Copy GPU-only params to device
    cudaMemcpy(d_shuffledWorkgroups, h_shuffledWorkgroups, ...);
    cudaMemcpy(d_scratchLocations, h_scratchLocations, ...);
    cudaMemcpy(d_kernelParams, h_kernelParams, ...);

    // 4. Launch GPU kernel (asynchronous)
    litmus_test<<<numWorkgroups, stressParams.workgroupSize>>>(
      d_testLocations, d_readResults, d_shuffledWorkgroups,
      d_gpuBarrier, d_scratchpad, d_scratchLocations,
      d_kernelParams, d_testInstances, d_hetBarriers);

    // 5. Launch CPU test threads
    bool doCpuPreStress = percentageCheck(stressParams.cpuPreStressPct);
    std::vector<std::thread> cpuTestThreads;
    for (int t = 0; t < numCpuThreads; t++) {
      int start = t * instancesPerCpuThread;
      int end = std::min(start + instancesPerCpuThread, testingThreads);
      if (start < testingThreads) {
        cpuTestThreads.emplace_back(cpu_test_thread,
          t, start, end,
          h_testLocations, h_readResults, h_kernelParams,
          h_hetBarriers, cpuScratchpad,
          doCpuPreStress, stressParams.cpuPreStressIterations,
          stressParams.cpuPreStressPattern);
      }
    }

    // 6. Join CPU threads + sync GPU
    for (auto& t : cpuTestThreads) t.join();
    cudaDeviceSynchronize();

    // 7. Run check_results kernel
    check_results<<<stressParams.testingWorkgroups, stressParams.workgroupSize>>>(
      d_testLocations, d_readResults, d_testResults, d_kernelParams, d_weak);
    cudaDeviceSynchronize();

    // 8. Copy results back and accumulate
    // For MEM_HOSTALLOC/MEM_MALLOC: h_testResults is already the host pointer
    // For MEM_MANAGED: need cudaDeviceSynchronize() first (already done)
    weakBehaviors += host_check_results(h_testResults, print_results);
    totalBehaviors += total_behaviors(h_testResults);
  }

  // 9. Stop stress threads, cleanup
  stopStress = true;
  for (auto& t : stressThreads) t.join();
  // het_free(...) for all shared allocations
  // cudaFree(...) for GPU-only allocations
}
```

### CPU Test Thread Function Pattern

This is defined in each kernel file (test-specific), but follows this pattern:

```cpp
void cpu_test_thread(
  int thread_id, int start_instance, int end_instance,
  het_atomic_uint* test_locations, ReadResults* read_results,
  KernelParams* kernel_params, het_barrier_t* het_barriers,
  uint* cpu_scratchpad,
  bool cpu_pre_stress, int cpu_pre_stress_iterations, int cpu_pre_stress_pattern) {

  for (int i = start_instance; i < end_instance; i++) {
    // Compute memory addresses for instance i (same formula as GPU)
    uint total_ids = kernel_params->total_instances;
    uint x_i = i * kernel_params->mem_stride * NUM_MEM_LOCS;
    uint y_i = cpu_permute_id(i, kernel_params->permute_location, total_ids)
               * kernel_params->mem_stride * NUM_MEM_LOCS + kernel_params->mem_offset;

    // Optional CPU pre-stress
    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    // Hit per-instance barrier (wait for GPU side of the logical instance)
    cpu_het_spin(&het_barriers[i], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

    // Execute CPU-side test operations for instance i
    // (test-specific — e.g., for MP with HET_C0_G1, Thread 0 does: store x, store y)
#ifdef THREAD_0_IS_CPU
    test_locations[x_i].store(1, cuda::memory_order_relaxed);
    test_locations[y_i].store(1, cuda::memory_order_release);
#endif
    // (other thread roles handled by #ifdef THREAD_N_IS_CPU blocks)
  }
}
```

### Helper Functions (from cuda-litmus, unchanged)

```cpp
int setBetween(int min, int max);
bool percentageCheck(int percentage);
void setShuffledWorkgroups(uint* h_shuffledWorkgroups, int numWorkgroups, int shufflePct);
void setScratchLocations(uint* h_locations, int numWorkgroups, StressParams params);
void setDynamicKernelParams(KernelParams* h_kernelParams, StressParams params);
void setStaticKernelParams(KernelParams* h_kernelParams, StressParams sp, TestParams tp);
int total_behaviors(TestResults* results);
int parseTestParamsFile(const char* filename, TestParams* config);
int parseStressParamsFile(const char* filename, StressParams* config);
```

---

## 6. Kernel File Pattern

Each kernel file (e.g., `kernels/mp.cu`) contains three things:

### a) `litmus_test` GPU kernel

The GPU kernel only executes the GPU-side thread roles. CPU-side roles are no-ops on the GPU.

**Example: MP (Message Passing) with HET_C0_G1**

```cpp
#include "litmus_het.cuh"
#include "functions.cu"

__global__ void litmus_test(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* gpu_barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances,
  het_barrier_t* het_barriers) {

  uint shuffled_workgroup = shuffled_workgroups[blockIdx.x];
  if (shuffled_workgroup < kernel_params->testing_workgroups) {
    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint y_0 = (wg_offset + permute_id_0) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (true) {
#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = test_locations[y_1].load(cuda::memory_order_acquire);
      uint r1 = test_locations[x_1].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_1].r1 = r1;
#endif

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      test_locations[y_0].store(1, cuda::memory_order_release);
#endif
    }
  }
  MEM_STRESS();
}
```

The current implementation collapses all role IDs for a heterogeneous logical instance onto one `logical_id` in `DEFINE_IDS()`:

```cpp
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint logical_id = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint id_0 = logical_id; \
  uint id_1 = logical_id; \
  uint id_2 = logical_id; \
  uint id_3 = logical_id; \
  uint wg_offset = 0;
```

That logical-instance model is what allows CPU and GPU participants to compute the same addresses and synchronize on `het_barriers[logical_id]`.

### b) `check_results` GPU kernel

Identical to cuda-litmus — classifies each instance's read results into outcome buckets. No het-specific changes needed because it reads from the shared `read_results` array that was populated by both CPU and GPU threads.

### c) `host_check_results` function

Identical to cuda-litmus — prints results and returns weak behavior count.

### d) `cpu_test_thread` function

The CPU thread entry point, defined in each kernel file because the test operations are test-specific.

```cpp
void cpu_test_thread(
  int thread_id, int start_instance, int end_instance,
  het_atomic_uint* test_locations, ReadResults* read_results,
  KernelParams* kernel_params, het_barrier_t* het_barriers,
  uint* cpu_scratchpad,
  bool cpu_pre_stress, int cpu_pre_stress_iterations, int cpu_pre_stress_pattern) {

  uint total_ids = kernel_params->total_instances;

  for (int inst = start_instance; inst < end_instance; inst++) {
    // Compute memory addresses for this instance
    // The CPU thread needs the same address computation as the GPU thread
    // For the thread role it's handling
    uint x_addr = inst * kernel_params->mem_stride * 2;
    uint y_addr = cpu_permute_id(inst, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    // Optional pre-stress
    if (cpu_pre_stress) {
      cpu_do_stress((volatile uint*)cpu_scratchpad,
                    cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    // Hit per-instance barrier
    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

    // Execute CPU-side operations
#ifdef THREAD_0_IS_CPU
    // MP Thread 0: store x=1 (relaxed), store y=1 (release)
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
    test_locations[y_addr].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_1_IS_CPU
    // MP Thread 1: load y (acquire), load x (relaxed)
    uint r0 = test_locations[y_addr].load(cuda::memory_order_acquire);
    uint r1 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif
  }
}
```

### Critical Implementation Detail: Logical-Instance Mapping

The current heterogeneous implementation uses a logical-instance model rather than cuda-litmus's original per-role cross-workgroup aliasing.

- `DEFINE_IDS()` collapses all role IDs for a heterogeneous instance to one `logical_id`
- CPU threads iterate over the same logical instance IDs directly
- Both sides compute addresses from that same logical instance ID
- `het_barriers[logical_id]` is the synchronization point for that instance

This keeps CPU and GPU participants aligned on a single memory-location tuple for each logical test instance.

---

## 7. tune.sh — Tuning Script

Adapted from cuda-litmus's `tune.sh` with additional dimensions for HET splits and `SCOPE_*`.

### Tuning file format (extended)

```
<test_name> <param_file>
<TB configs>
<HET split configs>
<SCOPE options>
<non-fence variants>
[<FENCE_SCOPE options>]
[<fence variants>]
```

Example for MP:
```
mp 2-loc.txt
TB_0_1 TB_01
HET_C0_G1 HET_C1_G0
SCOPE_SYSTEM SCOPE_DEVICE
RELAXED ACQ_REL
```

### Compilation command

```bash
/usr/local/cuda-12.4/bin/nvcc -D$tb -D$het -D$scope -D$variant -D$mem_backend \
     -I. -rdc=true -arch sm_90 \
     runner.cu "kernels/$test.cu" \
     -o "$TARGET_DIR/$test-$tb-$het-$scope-$fence_scope-$variant-$mem_short-runner"
```

For exhaustive compile/run coverage outside the interactive session, use `run-full-matrix.sh`. It consumes the same tuning files but writes compile logs, run logs, and CSV summaries under `full-matrix-results/`.

### Random config generation (extended)

Same as cuda-litmus's `random_config` function, plus:
```bash
echo "cpuStressThreads=$(random_between 0 16)" >> $PARAM_FILE
echo "cpuPreStressPct=$(random_between 0 100)" >> $PARAM_FILE
echo "cpuPreStressIterations=$(random_between 0 128)" >> $PARAM_FILE
echo "cpuPreStressPattern=$(random_between 0 3)" >> $PARAM_FILE
echo "barrierSpinLimit=$(random_between 1024 8192)" >> $PARAM_FILE
```

---

## 8. run-full-matrix.sh — Offline Matrix Runner

This helper script walks `all-tests.txt`, expands every tuning-file combination, compiles binaries, optionally runs them, and logs each outcome. It is intended for long-running offline verification on GH200.

Key behaviors:
- defaults to `MEM_MALLOC`, `/usr/local/cuda-12.4/bin/nvcc`, and `sm_90`
- supports `compile-only`, `run-only`, and combined `compile-and-run` modes
- `--tests` filters the matrix by one or more test families while `--tests-file` chooses the tuning inventory file
- `--resume <results-dir>` reuses the saved manifest and status files to continue an interrupted matrix run
- writes `compile.log`, `run.log`, `compile-summary.csv`, and `run-summary.csv`
- does not require modifying `tune.sh`

---

## 9. analyze.py — Analysis Script

Adapted from cuda-litmus's `analyze.py`. Reads `results/` directory and generates summary tables:
- Per-test weak behavior rates
- Best configurations per test
- Cross-test comparison (which splits/scopes/variants produce weak behaviors)
- Statistical analysis (mean, max, std of weak behavior rates)

---

## 10. Parameter Files

### params/2-loc.txt
```
numMemLocations=2
permuteLocation=1031
```

### params/3-loc.txt
```
numMemLocations=3
permuteLocation=1031
```

### params/4-loc.txt
```
numMemLocations=4
permuteLocation=1031
```

### params-smoke.txt
```
testIterations=2
testingWorkgroups=4
maxWorkgroups=4
workgroupSize=32
shufflePct=0
barrierPct=0
stressLineSize=64
stressTargetLines=1
scratchMemorySize=2048
memStride=1
memStressPct=0
memStressIterations=0
memStressPattern=0
preStressPct=0
preStressIterations=0
preStressPattern=0
stressAssignmentStrategy=0
permuteThread=419
cpuStressThreads=0
cpuPreStressPct=0
cpuPreStressIterations=0
cpuPreStressPattern=0
barrierSpinLimit=8192
```

---

## 11. Tuning Files

One per test. Format described in Section 7. See TEST-CATALOG.md for the full list of HET splits, TB configs, scopes, and variants per test.
