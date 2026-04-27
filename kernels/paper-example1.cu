// kernels/paper-example1.cu — Paper Example 1 (het)
// Thread 0: FENCE_0(), write x=1, write y=1
// Thread 1: r0 = read y, FENCE_1(), write z=1
// Thread 2: FENCE_2(), write z=2, write a=1
// Thread 3: r1 = read a, r2 = read x
// Weak behavior: r0==1 && z==2 && r1==1 && r2==0
// 4 memory locations: x, y, z, a (mem_stride * 4)
// DISALLOWED variant: acq/rel + seq_cst fences; default: all relaxed

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

#ifdef DISALLOWED
    cuda::memory_order load_order = cuda::memory_order_acquire;
    cuda::memory_order store_order = cuda::memory_order_release;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_seq_cst, FENCE_SCOPE);
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_seq_cst, FENCE_SCOPE);
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_seq_cst, FENCE_SCOPE);
#else
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    cuda::memory_order store_order = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2()
#endif

    DEFINE_IDS();

    // Thread 0: x, y
    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 4;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;

    // Thread 1: y, z
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint z_1 = (wg_offset + permute_id(permute_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    // Thread 2: z, a
    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint permute_permute_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint z_2 = (wg_offset + permute_permute_id_2) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint a_2 = (wg_offset + permute_id(permute_permute_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    // Thread 3: a, x
    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint permute_permute_id_3 = permute_id(permute_id_3, kernel_params->permute_location, total_ids);
    uint a_3 = (wg_offset + permute_id(permute_permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 4;

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      FENCE_0();
      test_locations[x_0].store(1, store_order); // write x
      test_locations[y_0].store(1, store_order); // write y
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = test_locations[y_1].load(load_order); // read y
      FENCE_1();
      test_locations[z_1].store(1, store_order); // write z
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      FENCE_2();
      test_locations[z_2].store(2, store_order); // write z=2
      test_locations[a_2].store(1, store_order); // write a
#endif

#ifdef THREAD_3_IS_GPU
      het_spin(&het_barriers[wg_offset + id_3], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r1 = test_locations[a_3].load(load_order); // read a
      uint r2 = test_locations[x_3].load(load_order); // read x
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
  uint total_ids = blockDim.x * kernel_params->testing_workgroups;
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 4];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  // Read z value via permute chain
  uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
  uint z_loc = permute_id(permute_id_0, kernel_params->permute_location, total_ids) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
  uint z = test_locations[z_loc];

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 1 && z == 2 && r1 == 1 && r2 == 0) { // weak behavior
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, z=2, r1=1, r2=0 (weak): " << results->weak << "\n";
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
    // Compute memory addresses (4 locations: x, y, z, a)
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
    // Thread 0: FENCE_0(), write x=1, write y=1
#ifdef DISALLOWED
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    test_locations[x_addr].store(1, cuda::memory_order_release);
    test_locations[y_addr].store(1, cuda::memory_order_release);
#else
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: r0 = read y, FENCE_1(), write z=1
#ifdef DISALLOWED
    uint r0 = test_locations[y_addr].load(cuda::memory_order_acquire);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    test_locations[z_addr].store(1, cuda::memory_order_release);
#else
    uint r0 = test_locations[y_addr].load(cuda::memory_order_relaxed);
    test_locations[z_addr].store(1, cuda::memory_order_relaxed);
#endif
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: FENCE_2(), write z=2, write a=1
#ifdef DISALLOWED
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    test_locations[z_addr].store(2, cuda::memory_order_release);
    test_locations[a_addr].store(1, cuda::memory_order_release);
#else
    test_locations[z_addr].store(2, cuda::memory_order_relaxed);
    test_locations[a_addr].store(1, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_3_IS_CPU
    // Thread 3: r1 = read a, r2 = read x
#ifdef DISALLOWED
    uint r1 = test_locations[a_addr].load(cuda::memory_order_acquire);
    uint r2 = test_locations[x_addr].load(cuda::memory_order_acquire);
#else
    uint r1 = test_locations[a_addr].load(cuda::memory_order_relaxed);
    uint r2 = test_locations[x_addr].load(cuda::memory_order_relaxed);
#endif
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r1 = r1;
    read_results[inst].r2 = r2;
#endif
  }
}
