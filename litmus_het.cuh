// litmus_het.cuh — Core header for heterogeneous CPU-GPU litmus testing
#ifndef LITMUS_HET_CUH
#define LITMUS_HET_CUH

#include <cuda_runtime.h>
#include <cuda/atomic>

// =============================================================================
// Atomic Type Definitions
// =============================================================================

// GPU-side test operations use the configured scope while the backing storage
// remains shared with CPU-side system-scope accesses.
#ifdef SCOPE_BLOCK
  #define TEST_SCOPE cuda::thread_scope_block
#elif defined(SCOPE_DEVICE)
  #define TEST_SCOPE cuda::thread_scope_device
#elif defined(SCOPE_SYSTEM)
  #define TEST_SCOPE cuda::thread_scope_system
#else
  #define TEST_SCOPE cuda::thread_scope_system
#endif

#define HET_BARRIER_EXPECTED 2

struct het_atomic_uint {
  uint value;

  template <cuda::thread_scope Scope>
  __host__ __device__ cuda::atomic_ref<uint, Scope> ref() {
    return cuda::atomic_ref<uint, Scope>(value);
  }

  template <cuda::thread_scope Scope>
  __host__ __device__ cuda::atomic_ref<const uint, Scope> ref() const {
    return cuda::atomic_ref<const uint, Scope>(value);
  }

  __host__ __device__ void store(uint desired,
                                 cuda::memory_order order = cuda::memory_order_seq_cst) {
#ifdef __CUDA_ARCH__
    ref<TEST_SCOPE>().store(desired, order);
#else
    ref<cuda::thread_scope_system>().store(desired, order);
#endif
  }

  __host__ __device__ uint load(cuda::memory_order order = cuda::memory_order_seq_cst) const {
#ifdef __CUDA_ARCH__
    return ref<TEST_SCOPE>().load(order);
#else
    return ref<cuda::thread_scope_system>().load(order);
#endif
  }

  __host__ __device__ operator uint() const {
    return load(cuda::memory_order_relaxed);
  }

  __host__ __device__ uint& raw() {
    return value;
  }

  __host__ __device__ const uint& raw() const {
    return value;
  }
};

// Per-instance barrier for CPU-GPU synchronization
typedef cuda::atomic<uint, cuda::thread_scope_system> het_barrier_t;

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

// =============================================================================
// HET Split Macros — Which thread roles run on CPU vs GPU
// =============================================================================

// --- 2-thread splits ---
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

// --- 3-thread splits (6 valid: at least 1 CPU + 1 GPU) ---
#ifdef HET_C0_G1_G2
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 2
#elif defined(HET_C1_G0_G2)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 2
#elif defined(HET_C2_G0_G1)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 2
#elif defined(HET_C0_C1_G2)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 1
#elif defined(HET_C0_C2_G1)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 1
#elif defined(HET_C1_C2_G0)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_CPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 1
#endif

// --- 4-thread splits (14 valid: at least 1 CPU + 1 GPU) ---
// 1 CPU, 3 GPU (4 combinations)
#ifdef HET_C0_G1_G2_G3
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 3
#elif defined(HET_C1_G0_G2_G3)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 3
#elif defined(HET_C2_G0_G1_G3)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 3
#elif defined(HET_C3_G0_G1_G2)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 1
  #define NUM_GPU_ROLES 3
#endif

// 2 CPU, 2 GPU (6 combinations)
#ifdef HET_C0_C1_G2_G3
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#elif defined(HET_C0_C2_G1_G3)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#elif defined(HET_C0_C3_G1_G2)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#elif defined(HET_C1_C2_G0_G3)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#elif defined(HET_C1_C3_G0_G2)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#elif defined(HET_C2_C3_G0_G1)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 2
  #define NUM_GPU_ROLES 2
#endif

// 3 CPU, 1 GPU (4 combinations)
#ifdef HET_C0_C1_C2_G3
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_GPU
  #define NUM_CPU_ROLES 3
  #define NUM_GPU_ROLES 1
#elif defined(HET_C0_C1_C3_G2)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_GPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 3
  #define NUM_GPU_ROLES 1
#elif defined(HET_C0_C2_C3_G1)
  #define THREAD_0_IS_CPU
  #define THREAD_1_IS_GPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 3
  #define NUM_GPU_ROLES 1
#elif defined(HET_C1_C2_C3_G0)
  #define THREAD_0_IS_GPU
  #define THREAD_1_IS_CPU
  #define THREAD_2_IS_CPU
  #define THREAD_3_IS_CPU
  #define NUM_CPU_ROLES 3
  #define NUM_GPU_ROLES 1
#endif

// =============================================================================
// TB_ Macros — GPU Thread ID Computation
// Identical to cuda-litmus. Only GPU threads use these.
// =============================================================================

#if defined(HET_C0_G1) || defined(HET_C1_G0) || \
    defined(HET_C0_G1_G2) || defined(HET_C1_G0_G2) || defined(HET_C2_G0_G1) || \
    defined(HET_C0_C1_G2) || defined(HET_C0_C2_G1) || defined(HET_C1_C2_G0) || \
    defined(HET_C0_G1_G2_G3) || defined(HET_C1_G0_G2_G3) || defined(HET_C2_G0_G1_G3) || defined(HET_C3_G0_G1_G2) || \
    defined(HET_C0_C1_G2_G3) || defined(HET_C0_C2_G1_G3) || defined(HET_C0_C3_G1_G2) || \
    defined(HET_C1_C2_G0_G3) || defined(HET_C1_C3_G0_G2) || defined(HET_C2_C3_G0_G1) || \
    defined(HET_C0_C1_C2_G3) || defined(HET_C0_C1_C3_G2) || defined(HET_C0_C2_C3_G1) || defined(HET_C1_C2_C3_G0)

#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint logical_id = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint id_0 = logical_id; \
  uint id_1 = logical_id; \
  uint id_2 = logical_id; \
  uint id_3 = logical_id; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#else

// --- 2-thread TB configurations ---
#ifdef TB_0_1
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_01)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;

#define RESULT_IDS() \
  uint total_ids = blockDim.x; \
  uint wg_offset = blockIdx.x * blockDim.x;
#else
// no 2-thread inclusion
#endif

// --- 3-thread TB configurations ---
#ifdef TB_0_1_2
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint third_workgroup = stripe_workgroup(new_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = third_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_01_2)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint id_1 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = new_workgroup * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_0_12)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint id_2 = new_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_02_1)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = new_workgroup * blockDim.x + threadIdx.x; \
  uint id_2 = shuffled_workgroup * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#define RESULT_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint wg_offset = 0;

#elif defined(TB_012)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = permute_id(id_1, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;

#define RESULT_IDS() \
  uint total_ids = blockDim.x; \
  uint wg_offset = blockIdx.x * blockDim.x;
#else
// no 3-thread inclusion
#endif

// --- 4-thread TB configurations ---
#ifdef TB_0_1_2_3
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_01_2_3)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_01_23)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_1_23)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_02_1_3)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_02_13)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_2_13)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_03_1_2)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = shuffled_workgroup; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_03_12)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x * kernel_params->testing_workgroups; \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_1 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = shuffled_workgroup; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0_12_3)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_0_123)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = workgroup_1; \
  uint id_2_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = workgroup_2 * blockDim.x + id_2_local; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_2_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_012_3)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_1 = workgroup_1 * blockDim.x + id_1_local; \
  uint workgroup_2 = workgroup_1; \
  uint id_2 = workgroup_2 * blockDim.x + permute_id(id_1_local, kernel_params->permute_thread, blockDim.x); \
  uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_3 = workgroup_3 * blockDim.x + threadIdx.x; \
  uint wg_offset = 0;

#elif defined(TB_023_1)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_1 = workgroup_1 * blockDim.x + threadIdx.x; \
  uint workgroup_2 = shuffled_workgroup; \
  uint id_2_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = workgroup_2 * blockDim.x + id_2_local; \
  uint workgroup_3 = workgroup_2; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_2_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_013_2)
#define DEFINE_IDS() \
  uint total_ids = (blockDim.x * kernel_params->testing_workgroups); \
  uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x; \
  uint workgroup_1 = shuffled_workgroup; \
  uint id_1_local = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_1 = workgroup_1 * blockDim.x + id_1_local; \
  uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups); \
  uint id_2 = workgroup_2 * blockDim.x + threadIdx.x; \
  uint workgroup_3 = workgroup_1; \
  uint id_3 = workgroup_3 * blockDim.x + permute_id(id_1_local, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = 0;

#elif defined(TB_0123)
#define DEFINE_IDS() \
  uint total_ids = blockDim.x; \
  uint id_0 = threadIdx.x; \
  uint id_1 = permute_id(threadIdx.x, kernel_params->permute_thread, blockDim.x); \
  uint id_2 = permute_id(id_1, kernel_params->permute_thread, blockDim.x); \
  uint id_3 = permute_id(id_2, kernel_params->permute_thread, blockDim.x); \
  uint wg_offset = shuffled_workgroup * blockDim.x;
#else
// no 4-thread inclusion
#endif

#endif

// =============================================================================
// Memory Location Macros
// =============================================================================

#define TWO_THREAD_TWO_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2; \
  uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids); \
  uint y_0 = (wg_offset + permute_id_0) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2; \
  uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids); \
  uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

#define THREE_THREAD_TWO_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2; \
  uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2; \
  uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 2; \
  uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset; \
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x; \
  test_instances[id_0].t0 = t_id; \
  test_instances[id_1].t1 = t_id; \
  test_instances[id_2].t2 = t_id; \
  test_instances[id_0].x = x_0; \
  test_instances[id_0].y = y_0;

#define THREE_THREAD_THREE_MEM_LOCATIONS() \
  uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 3; \
  uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids); \
  uint y_0 = (wg_offset + permute_id_0) * kernel_params->mem_stride * 3 + kernel_params->mem_offset; \
  uint z_0 = (wg_offset + permute_id(permute_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids); \
  uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 3 + kernel_params->mem_offset; \
  uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 3; \
  uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids); \
  uint z_2 = (wg_offset + permute_id(permute_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset; \
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x; \
  test_instances[id_0].t0 = t_id; \
  test_instances[id_1].t1 = t_id; \
  test_instances[id_2].t2 = t_id; \
  test_instances[id_0].x = x_0; \
  test_instances[id_0].y = y_0; \
  test_instances[id_0].z = z_0;

// =============================================================================
// Stress Macros (GPU side)
// =============================================================================

#define PRE_STRESS() \
  if (kernel_params->pre_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern); \
  } \
  if (kernel_params->barrier) { \
    spin(gpu_barrier, blockDim.x * kernel_params->testing_workgroups); \
  }

#define MEM_STRESS() \
  else if (kernel_params->mem_stress) { \
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->mem_stress_pattern); \
  }

// =============================================================================
// Data Structures
// =============================================================================

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

// =============================================================================
// Function Declarations
// =============================================================================

// GPU kernel — litmus test (runs GPU-side thread operations)
__global__ void litmus_test(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  uint* shuffled_workgroups,
  cuda::atomic<uint, cuda::thread_scope_device>* gpu_barrier,
  uint* scratchpad,
  uint* scratch_locations,
  KernelParams* kernel_params,
  TestInstance* test_instances,
  het_barrier_t* het_barriers);

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
  volatile uint* cpu_scratchpad,
  bool cpu_pre_stress,
  int cpu_pre_stress_iterations,
  int cpu_pre_stress_pattern);

#endif // LITMUS_HET_CUH
