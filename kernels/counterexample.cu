// kernels/counterexample.cu — CTA Scope Mismatch Counterexample (het)
// Thread 0: write z=1 (seq_cst, sys scope)
// Thread 1: read z (seq_cst, CTA scope), write x=1 (seq_cst, CTA scope)
// Thread 2: write x=2 (seq_cst, sys scope), write y=1 (release, sys scope)
// Thread 3: read y (acquire, sys scope), read z (seq_cst, sys scope)
// Weak behavior: r0==1 && x==2 && r1==1 && r2==0
// 3 memory locations: x, y, z (mem_stride * 3)
// No variant macros — fixed memory orders with CTA-scoped pointer casts

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

    DEFINE_IDS();

    // Memory locations for each thread
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint z_0 = (wg_offset + permute_id(permute_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 3;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint x_2 = (wg_offset + id_2) * kernel_params->mem_stride * 3;
    uint y_2 = (wg_offset + permute_id_2) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;

    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint y_3 = (wg_offset + permute_id_3) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_3 = (wg_offset + permute_id(permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    // CTA-scoped pointer casts for Thread 1
    cuda::atomic<uint, cuda::thread_scope_block>* z_1_ptr = (cuda::atomic<uint, cuda::thread_scope_block>*) &test_locations[z_1];
    cuda::atomic<uint, cuda::thread_scope_block>* x_1_ptr = (cuda::atomic<uint, cuda::thread_scope_block>*) &test_locations[x_1];

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[z_0].store(1, cuda::memory_order_seq_cst); // write z, sys scope
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = z_1_ptr->load(cuda::memory_order_seq_cst); // read z, CTA scope
      x_1_ptr->store(1, cuda::memory_order_seq_cst); // write x=1, CTA scope
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[x_2].store(2, cuda::memory_order_seq_cst); // write x=2, sys scope
      test_locations[y_2].store(1, cuda::memory_order_release); // write y=1, sys scope
#endif

#ifdef THREAD_3_IS_GPU
      het_spin(&het_barriers[wg_offset + id_3], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r1 = test_locations[y_3].load(cuda::memory_order_acquire); // read y, sys scope
      uint r2 = test_locations[z_3].load(cuda::memory_order_seq_cst); // read z, sys scope
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_3].r1 = r1;
      read_results[wg_offset + id_3].r2 = r2;
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
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && x == 2 && r1 == 1 && r2 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, x=2, r1=1, r2=0 (weak): " << results->weak << "\n";
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
    // Compute memory addresses (3 locations: x, y, z)
    uint permute_0 = cpu_permute_id(inst, kernel_params->permute_location, total_ids);
    uint permute_2_0 = cpu_permute_id(permute_0, kernel_params->permute_location, total_ids);

    uint x_addr = inst * kernel_params->mem_stride * 3;
    uint y_addr = permute_0 * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_addr = permute_2_0 * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    // Thread 0: write z=1 (seq_cst, sys scope)
    test_locations[z_addr].store(1, cuda::memory_order_seq_cst);
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: read z (seq_cst), write x=1 (seq_cst)
    // Note: CTA scope is GPU-only concept; CPU always uses system scope
    uint r0 = test_locations[z_addr].load(cuda::memory_order_seq_cst);
    test_locations[x_addr].store(1, cuda::memory_order_seq_cst);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: write x=2 (seq_cst), write y=1 (release)
    test_locations[x_addr].store(2, cuda::memory_order_seq_cst);
    test_locations[y_addr].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_3_IS_CPU
    // Thread 3: read y (acquire), read z (seq_cst)
    uint r1 = test_locations[y_addr].load(cuda::memory_order_acquire);
    uint r2 = test_locations[z_addr].load(cuda::memory_order_seq_cst);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r1 = r1;
    read_results[inst].r2 = r2;
#endif
  }
}
