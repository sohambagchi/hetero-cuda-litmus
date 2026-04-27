// kernels/paper-example.cu — Paper Example (het)
// Thread 0: fence(seq_cst, device), write x=1 (release), write z=1 (release)
// Thread 1: read z (acquire), write y=1 (release), fence(seq_cst, device), read y (relaxed)
// Thread 2: fence(seq_cst, device), write y=2 (release), write a=1 (release)
// Thread 3: read a (acquire), read x (acquire), fence(seq_cst, device)
// Weak behavior: r0==1 && r1==2 && r2==1 && r3==0
// 4 memory locations: x, y, z, a (mem_stride * 4)
// No variant macros — hardcoded device-scope fences
// NOTE: Original uses TB_0_1_2_3 hardcoded. Het version uses DEFINE_IDS().

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

    // Hardcode TB_0_1_2_3 topology (matching original)
    uint total_ids = blockDim.x * kernel_params->testing_workgroups;
    uint id_0 = shuffled_workgroup * blockDim.x + threadIdx.x;
    uint workgroup_1 = stripe_workgroup(shuffled_workgroup, threadIdx.x, kernel_params->testing_workgroups);
    uint id_1 = workgroup_1 * blockDim.x + threadIdx.x;
    uint workgroup_2 = stripe_workgroup(workgroup_1, threadIdx.x, kernel_params->testing_workgroups);
    uint id_2 = workgroup_2 * blockDim.x + threadIdx.x;
    uint workgroup_3 = stripe_workgroup(workgroup_2, threadIdx.x, kernel_params->testing_workgroups);
    uint id_3 = workgroup_3 * blockDim.x + threadIdx.x;
    uint wg_offset = 0;

    // Memory addresses (original uses no wg_offset in address calc)
    uint x_0 = id_0 * kernel_params->mem_stride * 4;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint z_0 = permute_id(permute_id_0, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = permute_id_1 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint z_1 = permute_id(permute_id_1, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint y_2 = permute_id_2 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_permute_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint a_2 = permute_id(permute_permute_id_2, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    uint x_3 = id_3 * kernel_params->mem_stride * 4;
    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint permute_permute_id_3 = permute_id(permute_id_3, kernel_params->permute_location, total_ids);
    uint a_3 = permute_id(permute_permute_id_3, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      test_locations[x_0].store(1, cuda::memory_order_release); // write x
      test_locations[z_0].store(1, cuda::memory_order_release); // write z
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = test_locations[z_1].load(cuda::memory_order_acquire); // read z
      test_locations[y_1].store(1, cuda::memory_order_release); // write y
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed); // read y
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_1].r0 = r0;
      read_results[id_1].r1 = r1;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      test_locations[y_2].store(2, cuda::memory_order_release); // write y=2
      test_locations[a_2].store(1, cuda::memory_order_release); // write a
#endif

#ifdef THREAD_3_IS_GPU
      het_spin(&het_barriers[id_3], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r2 = test_locations[a_3].load(cuda::memory_order_acquire); // read a
      uint r3 = test_locations[x_3].load(cuda::memory_order_acquire); // read x
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[id_3].r2 = r2;
      read_results[id_3].r3 = r3;
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
  uint x = test_locations[id_0 * kernel_params->mem_stride * 4];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint r3 = read_results[id_0].r3;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && r1 == 2 && r2 == 1 && r3 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=2, r2=1, r3=0 (weak): " << results->weak << "\n";
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
    // Compute memory addresses (4 locations, same formulas as GPU, no wg_offset)
    uint permute_0 = cpu_permute_id(inst, kernel_params->permute_location, total_ids);
    uint permute_2_0 = cpu_permute_id(permute_0, kernel_params->permute_location, total_ids);

    uint x_addr = inst * kernel_params->mem_stride * 4;
    uint y_addr = permute_0 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint z_addr = permute_2_0 * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint a_addr = cpu_permute_id(permute_2_0, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    // Thread 0: fence(seq_cst), write x=1 (release), write z=1 (release)
    // Note: device-scope fence on GPU — CPU uses system scope
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    test_locations[x_addr].store(1, cuda::memory_order_release);
    test_locations[z_addr].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: read z (acquire), write y=1 (release), fence(seq_cst), read y (relaxed)
    uint r0 = test_locations[z_addr].load(cuda::memory_order_acquire);
    test_locations[y_addr].store(1, cuda::memory_order_release);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    uint r1 = test_locations[y_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: fence(seq_cst), write y=2 (release), write a=1 (release)
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    test_locations[y_addr].store(2, cuda::memory_order_release);
    test_locations[a_addr].store(1, cuda::memory_order_release);
#endif

#ifdef THREAD_3_IS_CPU
    // Thread 3: read a (acquire), read x (acquire), fence(seq_cst)
    uint r2 = test_locations[a_addr].load(cuda::memory_order_acquire);
    uint r3 = test_locations[x_addr].load(cuda::memory_order_acquire);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r2 = r2;
    read_results[inst].r3 = r3;
#endif
  }
}
