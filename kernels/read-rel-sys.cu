// kernels/read-rel-sys.cu — Read Release System scope test (het)
// Thread 0: store x=1 (relaxed), store y=1 (release)
// Thread 1: store y=2 (relaxed), r0 = load x (relaxed)
// Weak behavior: r0==0 && y==2

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

    // No variants — hardcoded orderings (always inter-block TB_0_1)
    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint new_workgroup = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = new_workgroup * blockDim.x + threadIdx.x;
    uint wg_offset = 0;

    uint x_0 = id_0 * kernel_params->mem_stride * 2;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint y_0 = permute_id_0 * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    uint x_1 = id_1 * kernel_params->mem_stride * 2;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = permute_id_1 * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      test_locations[y_0].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[y_1].store(2, cuda::memory_order_relaxed);
      uint r0 = test_locations[x_1].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
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
  uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, gridDim.x * blockDim.x);
  uint y = test_locations[permute_id_0 * kernel_params->mem_stride * 2 + kernel_params->mem_offset];
  uint r0 = read_results[id_0].r0;

  if (x == 0) {
    test_results->na.fetch_add(1);
  }
  else if (r0 == 0 && y == 2) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, y=2 (weak): " << results->weak << "\n";
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
    uint x_addr = inst * kernel_params->mem_stride * 2;
    uint y_addr = cpu_permute_id(inst, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
    test_locations[y_addr].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_1_IS_CPU
    test_locations[y_addr].store(2, cuda::memory_order_relaxed);
    uint r0 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
#endif
  }
}
