// kernels/2+2w.cu — Two Plus Two Writes (het)
// Thread 0: store x=1 (relaxed), FENCE0(), store y=2 (store_order0)
// Thread 1: store y=1 (relaxed), FENCE1(), store x=2 (store_order1)
// Weak behavior: x==1 && y==1 (both early stores "win" coherence)

#include <iostream>
#include "../litmus_het.cuh"
#include "../cpu_functions.h"
#include "../functions.cu"

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

#ifdef RELEASE
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    #define FENCE0()
    #define FENCE1()
#elif defined(RELAXED)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
#elif defined(BOTH_FENCE)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_0)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1()
#elif defined(FENCE_1)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
#endif

    DEFINE_IDS();
    TWO_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 0: store x=1, fence, store y=2
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      FENCE0()
      test_locations[y_0].store(2, store_order0);
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 1: store y=1, fence, store x=2
      test_locations[y_1].store(1, cuda::memory_order_relaxed);
      FENCE1()
      test_locations[x_1].store(2, store_order1);
#endif
    }
  }
  MEM_STRESS();
}

__global__ void check_results(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  RESULT_IDS();
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[(wg_offset + id_0) * kernel_params->mem_stride * 2];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];

  if (x == 1 && y == 1) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else if (x == 0) {
    test_results->na.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "x=1, y=1 (weak): " << results->weak << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak;
}

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
  int cpu_pre_stress_pattern) {

  uint total_ids = kernel_params->total_instances;

  for (int inst = start_instance; inst < end_instance; inst++) {
    uint x_addr = inst * kernel_params->mem_stride * 2;
    uint y_addr = cpu_permute_id(inst, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    // Thread 0: store x=1, store y=2
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
    // CPU cannot do GPU FENCE macros; use atomic_thread_fence with system scope
#if defined(BOTH_FENCE) || defined(FENCE_0)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#ifdef RELEASE
    test_locations[y_addr].store(2, cuda::memory_order_release);
#else
    test_locations[y_addr].store(2, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: store y=1, store x=2
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#if defined(BOTH_FENCE) || defined(FENCE_1)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#ifdef RELEASE
    test_locations[x_addr].store(2, cuda::memory_order_release);
#else
    test_locations[x_addr].store(2, cuda::memory_order_relaxed);
#endif
#endif
  }
}
