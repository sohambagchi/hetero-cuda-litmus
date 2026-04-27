// kernels/iriw.cu — Independent Reads of Independent Writes (het)
// Thread 0: store x=1
// Thread 1: r0 = load x, FENCE_1(), r1 = load y
// Thread 2: store y=1
// Thread 3: r2 = load y, FENCE_3(), r3 = load x
// Weak behavior: r0==1 && r1==0 && r2==1 && r3==0

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
    cuda::memory_order thread_1_order = cuda::memory_order_acquire;
    cuda::memory_order thread_3_order = cuda::memory_order_acquire;
    #define FENCE_1()
    #define FENCE_3()
#elif defined(THREAD_1_FENCE)
    cuda::memory_order thread_1_order = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_order = cuda::memory_order_acquire;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_3()
#elif defined(THREAD_3_FENCE)
    cuda::memory_order thread_1_order = cuda::memory_order_acquire;
    cuda::memory_order thread_3_order = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_3() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(BOTH_FENCE)
    cuda::memory_order thread_1_order = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_order = cuda::memory_order_relaxed;
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_3() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(RELAXED)
    cuda::memory_order thread_1_order = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_order = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_3()
#else
    cuda::memory_order thread_1_order = cuda::memory_order_relaxed;
    cuda::memory_order thread_3_order = cuda::memory_order_relaxed;
    #define FENCE_1()
    #define FENCE_3()
#endif

    DEFINE_IDS();

    uint x_0 = (wg_offset + id_0) * kernel_params->mem_stride * 2;
    uint y_0 = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_1 = (wg_offset + id_1) * kernel_params->mem_stride * 2;
    uint y_1 = (wg_offset + permute_id(id_1, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint y_2 = (wg_offset + permute_id(id_2, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;
    uint x_3 = (wg_offset + id_3) * kernel_params->mem_stride * 2;
    uint y_3 = (wg_offset + permute_id(id_3, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 2 + kernel_params->mem_offset;

    uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
    test_instances[id_0].t0 = t_id;
    test_instances[id_1].t1 = t_id;
    test_instances[id_2].t2 = t_id;
    test_instances[id_3].t3 = t_id;
    test_instances[id_0].x = x_0;
    test_instances[id_0].y = y_0;

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[x_0].store(1, cuda::memory_order_relaxed); // write x
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r0 = test_locations[x_1].load(thread_1_order); // read x
      FENCE_1()
      uint r1 = test_locations[y_1].load(cuda::memory_order_relaxed); // read y
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
      read_results[wg_offset + id_1].r1 = r1;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      test_locations[y_2].store(1, cuda::memory_order_relaxed); // write y
#endif

#ifdef THREAD_3_IS_GPU
      het_spin(&het_barriers[wg_offset + id_3], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      uint r2 = test_locations[y_3].load(thread_3_order); // read y
      FENCE_3()
      uint r3 = test_locations[x_3].load(cuda::memory_order_relaxed); // read x
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
  uint x = test_locations[id_0 * kernel_params->mem_stride * 2];
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint r3 = read_results[id_0].r3;

  if (x == 0) {
    test_results->na.fetch_add(1); // thread skipped
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0) { // both observers run first
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1 && r3 == 1) { // both observers run last
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1 && r3 == 1) { // first observer runs first
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0 && r3 == 0) { // second observer runs first
    test_results->res3.fetch_add(1);
  }
  else if (r0 == r1 && r2 != r3) { // second observer interleaved
    test_results->res4.fetch_add(1);
  }
  else if (r0 != r1 && r2 == r3) { // first observer interleaved
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0 && r3 == 1) { // both interleaved
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 1 && r3 == 0) { // both interleaved
    test_results->res7.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0 && r3 == 1) { // both interleaved
    test_results->res8.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0) { // observer threads see x/y in different orders
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=0, r2=0, r3=0 (seq): " << results->res0 << "\n";
    std::cout << "r0=1, r1=1, r2=1, r3=1 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1, r3=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=1, r1=1, r2=0, r3=0 (seq): " << results->res3 << "\n";
    std::cout << "r0 == r1, r2 != r3 (seq/interleaved): " << results->res4 << "\n";
    std::cout << "r0 != r1, r2 == r3 (interleaved/seq): " << results->res5 << "\n";
    std::cout << "r0=0, r1=1, r2=0, r3=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=0, r1=1, r2=1, r3=0 (interleaved): " << results->res7 << "\n";
    std::cout << "r0=1, r1=0, r2=0, r3=1 (interleaved): " << results->res8 << "\n";
    std::cout << "r0=1, r1=0, r2=1, r3=0 (weak): " << results->weak << "\n";
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
    // Thread 0: store x=1
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: r0 = load x, fence, r1 = load y
#if defined(ACQUIRE)
    uint r0 = test_locations[x_addr].load(cuda::memory_order_acquire);
#else
    uint r0 = test_locations[x_addr].load(cuda::memory_order_relaxed);
#endif
#if defined(THREAD_1_FENCE) || defined(BOTH_FENCE)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
    uint r1 = test_locations[y_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: store y=1
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#endif

#ifdef THREAD_3_IS_CPU
    // Thread 3: r2 = load y, fence, r3 = load x
#if defined(ACQUIRE)
    uint r2 = test_locations[y_addr].load(cuda::memory_order_acquire);
#else
    uint r2 = test_locations[y_addr].load(cuda::memory_order_relaxed);
#endif
#if defined(THREAD_3_FENCE) || defined(BOTH_FENCE)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
    uint r3 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r2 = r2;
    read_results[inst].r3 = r3;
#endif
  }
}
