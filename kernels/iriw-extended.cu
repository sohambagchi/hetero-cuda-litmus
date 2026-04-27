// kernels/iriw-extended.cu — IRIW Extended with a/b cross-links (het)
// Thread 0: write x=1, read a, read b
// Thread 1: read x, read y, write b=1
// Thread 2: write y=1, read b, read a
// Thread 3: read y, read x, write a=1
// Weak behavior (both): r0==1,r1==0,r2==1,r3==0 AND r4==1,r5==0,r6==1,r7==0
// 4 memory locations: x, y, a, b (mem_stride * 4)

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

#ifdef ACQUIRE
    cuda::memory_order load_order = cuda::memory_order_acquire;
#else
    cuda::memory_order load_order = cuda::memory_order_relaxed;
#endif

    DEFINE_IDS();

    // Thread 0: x, a, b
    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 4;
    uint permute_id_0 = permute_id(id_0, kernel_params->permute_location, total_ids);
    uint permute_2_id_0 = permute_id(permute_id_0, kernel_params->permute_location, total_ids);
    uint a_0 = (wg_offset + permute_2_id_0) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint b_0 = (wg_offset + permute_id(permute_2_id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    // Thread 1: x, y, b
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 4;
    uint permute_id_1 = permute_id(id_1, kernel_params->permute_location, total_ids);
    uint y_1 = (wg_offset + permute_id_1) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_2_id_1 = permute_id(permute_id_1, kernel_params->permute_location, total_ids);
    uint b_1 = (wg_offset + permute_id(permute_2_id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    // Thread 2: y, b, a
    uint permute_id_2 = permute_id(id_2, kernel_params->permute_location, total_ids);
    uint y_2 = (wg_offset + permute_id_2) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint permute_2_id_2 = permute_id(permute_id_2, kernel_params->permute_location, total_ids);
    uint a_2 = (wg_offset + permute_2_id_2) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint b_2 = (wg_offset + permute_id(permute_2_id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    // Thread 3: y, x, a
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 4;
    uint permute_id_3 = permute_id(id_3, kernel_params->permute_location, total_ids);
    uint y_3 = (wg_offset + permute_id_3) * kernel_params->mem_stride * 4 + kernel_params->mem_offset;
    uint a_3 = (wg_offset + permute_id(permute_id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[x_0].store(1, cuda::memory_order_relaxed); // write x
      uint r4 = test_locations[a_0].load(load_order); // read a
      uint r5 = test_locations[b_0].load(cuda::memory_order_relaxed); // read b
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_0].r4 = r4;
      read_results[wg_offset + id_0].r5 = r5;
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = test_locations[x_1].load(load_order); // read x
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed); // read y
      test_locations[b_1].store(1, cuda::memory_order_relaxed); // write b
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[y_2].store(1, cuda::memory_order_relaxed); // write y
      uint r6 = test_locations[b_2].load(load_order); // read b
      uint r7 = test_locations[a_2].load(cuda::memory_order_relaxed); // read a
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_2].r6 = r6;
      read_results[wg_offset + id_2].r7 = r7;
#endif

#ifdef THREAD_3_IS_GPU
      het_spin(&het_barriers[wg_offset + id_3], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r2 = test_locations[y_3].load(load_order); // read y
      uint r3 = test_locations[x_3].load(cuda::memory_order_relaxed); // read x
      test_locations[a_3].store(1, cuda::memory_order_relaxed); // write a
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_3].r2 = r2;
      read_results[wg_offset + id_3].r3 = r3;
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
  uint r4 = read_results[id_0].r4;
  uint r5 = read_results[id_0].r5;
  uint r6 = read_results[id_0].r6;
  uint r7 = read_results[id_0].r7;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
    return;
  }
  bool checked = false;
  if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // x/y weak
    test_results->res14.fetch_add(1);
    checked = true;
  }
  if (r4 == 1 && r5 == 0 && r6 == 1 && r7 == 0) { // a/b weak
    test_results->res15.fetch_add(1);
    checked = true;
  }
  if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0 && r4 == 1 && r5 == 0 && r6 == 1 && r7 == 0) { // both weak
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
    checked = true;
  }
  if (!checked) {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0, r2=1, r3=0, r4=1, r5=0, r6=1, r7=0 (both weak): " << results->weak << "\n";
    std::cout << "r0=1, r1=0, r2=1, r3=0 (x/y weak): " << results->res14 << "\n";
    std::cout << "r4=1, r5=0, r6=1, r7=0 (a/b weak): " << results->res15 << "\n";
    std::cout << "thread skipped: " << results->na << "\n";
    std::cout << "other: " << results->other << "\n";
  }
  return results->weak + results->res14 + results->res15;
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
    // Compute memory addresses (4 locations: x, y, a, b)
    uint permute_0 = cpu_permute_id(inst, kernel_params->permute_location, total_ids);
    uint permute_2_0 = cpu_permute_id(permute_0, kernel_params->permute_location, total_ids);

    // Thread 0 addresses: x, a, b
    uint x_addr = inst * kernel_params->mem_stride * 4;
    uint a_addr = permute_2_0 * kernel_params->mem_stride * 4 + 2 * kernel_params->mem_offset;
    uint b_addr = cpu_permute_id(permute_2_0, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 4 + 3 * kernel_params->mem_offset;

    // Thread 1 addresses: x, y, b
    uint y_addr = permute_0 * kernel_params->mem_stride * 4 + kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    // Thread 0: write x=1, read a, read b
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
#ifdef ACQUIRE
    uint r4 = test_locations[a_addr].load(cuda::memory_order_acquire);
#else
    uint r4 = test_locations[a_addr].load(cuda::memory_order_relaxed);
#endif
    uint r5 = test_locations[b_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r4 = r4;
    read_results[inst].r5 = r5;
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: read x, read y, write b=1
#ifdef ACQUIRE
    uint r0 = test_locations[x_addr].load(cuda::memory_order_acquire);
#else
    uint r0 = test_locations[x_addr].load(cuda::memory_order_relaxed);
#endif
    uint r1 = test_locations[y_addr].load(cuda::memory_order_relaxed);
    test_locations[b_addr].store(1, cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: write y=1, read b, read a
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#ifdef ACQUIRE
    uint r6 = test_locations[b_addr].load(cuda::memory_order_acquire);
#else
    uint r6 = test_locations[b_addr].load(cuda::memory_order_relaxed);
#endif
    uint r7 = test_locations[a_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r6 = r6;
    read_results[inst].r7 = r7;
#endif

#ifdef THREAD_3_IS_CPU
    // Thread 3: read y, read x, write a=1
#ifdef ACQUIRE
    uint r2 = test_locations[y_addr].load(cuda::memory_order_acquire);
#else
    uint r2 = test_locations[y_addr].load(cuda::memory_order_relaxed);
#endif
    uint r3 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    test_locations[a_addr].store(1, cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r2 = r2;
    read_results[inst].r3 = r3;
#endif
  }
}
