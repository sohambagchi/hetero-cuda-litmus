// kernels/mp.cu — Message Passing (het)
// Thread 0 (producer): store x=1 (relaxed), store y=1 (release)
// Thread 1 (consumer): r0 = load y (acquire), r1 = load x (relaxed)
// Weak behavior: r0==1 && r1==0

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

#ifdef ACQ_REL
    cuda::memory_order store_order = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
#elif defined(RELAXED)
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
#else
    // Default: release/acquire
    cuda::memory_order store_order = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
#endif

    DEFINE_IDS();
    TWO_THREAD_TWO_MEM_LOCATIONS();

    PRE_STRESS();

    if (true) {

      // GPU handles its assigned thread role(s)
      // For HET_C0_G1: Thread 0 is CPU, Thread 1 is GPU
      // For HET_C1_G0: Thread 0 is GPU, Thread 1 is CPU

#ifdef THREAD_0_IS_GPU
      // Het barrier: wait for CPU thread handling Thread 1
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);

      // Thread 0 (producer): store x=1, store y=1
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      test_locations[y_0].store(1, store_order);
#endif

#ifdef THREAD_1_IS_GPU
      // Het barrier: wait for CPU thread handling Thread 0
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);

      // Thread 1 (consumer): load y, load x
      uint r0 = test_locations[y_1].load(load_order);
      uint r1 = test_locations[x_1].load(cuda::memory_order_relaxed);

      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
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
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0 (weak): " << results->weak << "\n";
    std::cout << "other: " << results->other << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
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
    // Compute memory addresses (same formula as GPU)
    uint x_addr = inst * kernel_params->mem_stride * 2;
    uint y_addr = cpu_permute_id(inst, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    // Optional CPU pre-stress
    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    // Hit per-instance barrier (synchronize with GPU)
    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

    // Execute CPU-side operations
#ifdef THREAD_0_IS_CPU
    // Thread 0 (producer): store x=1 (relaxed), store y=1 (release)
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
#ifdef ACQ_REL
    test_locations[y_addr].store(1, cuda::memory_order_release);
#elif defined(RELAXED)
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#else
    test_locations[y_addr].store(1, cuda::memory_order_release);
#endif
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1 (consumer): load y, load x
#ifdef ACQ_REL
    uint r0 = test_locations[y_addr].load(cuda::memory_order_acquire);
#elif defined(RELAXED)
    uint r0 = test_locations[y_addr].load(cuda::memory_order_relaxed);
#else
    uint r0 = test_locations[y_addr].load(cuda::memory_order_acquire);
#endif
    uint r1 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif
  }
}
