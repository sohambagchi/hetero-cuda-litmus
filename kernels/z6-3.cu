// kernels/z6-3.cu — Write Serialization Z6.3 (het)
// Thread 0: store x=1, FENCE0(), store y=1
// Thread 1: store y=2, FENCE1(), store z=1
// Thread 2: r0 = load z, FENCE2(), r1 = load x
// Weak behavior: y==2 && r0==1 && r1==0

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
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#elif defined(RELAXED)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#elif defined(ALL_FENCE)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_0)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1()
    #define FENCE2()
#elif defined(FENCE_1)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0()
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2()
#elif defined(FENCE_2)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_01)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_acquire;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2()
#elif defined(FENCE_02)
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_release;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE1()
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(FENCE_12)
    cuda::memory_order store_order0 = cuda::memory_order_release;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order store_order0 = cuda::memory_order_relaxed;
    cuda::memory_order store_order1 = cuda::memory_order_relaxed;
    cuda::memory_order load_order = cuda::memory_order_relaxed;
    #define FENCE0()
    #define FENCE1()
    #define FENCE2()
#endif

    DEFINE_IDS();
    THREE_THREAD_THREE_MEM_LOCATIONS();

    PRE_STRESS();

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 0: store x=1, fence, store y=1
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      FENCE0()
      test_locations[y_0].store(1, store_order0);
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 1: store y=2, fence, store z=1
      test_locations[y_1].store(2, cuda::memory_order_relaxed);
      FENCE1()
      test_locations[z_1].store(1, store_order1);
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 2: r0 = load z, fence, r1 = load x
      uint r0 = test_locations[z_2].load(load_order);
      FENCE2()
      uint r1 = test_locations[x_2].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_2].r0 = r0;
      read_results[wg_offset + id_2].r1 = r1;
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
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];
  uint y_loc = (wg_offset + permute_id(id_0, kernel_params->permute_location, total_ids)) * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
  uint y = test_locations[y_loc];

  if (x == 0) {
    test_results->na.fetch_add(1);
  }
  else if (y == 2 && r0 == 1 && r1 == 0) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=1, r1=0, y=2 (weak): " << results->weak << "\n";
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
    uint x_addr = inst * kernel_params->mem_stride * 3;
    uint perm_inst = cpu_permute_id(inst, kernel_params->permute_location, total_ids);
    uint y_addr = perm_inst * kernel_params->mem_stride * 3 + kernel_params->mem_offset;
    uint z_addr = cpu_permute_id(perm_inst, kernel_params->permute_location, total_ids)
                  * kernel_params->mem_stride * 3 + 2 * kernel_params->mem_offset;

    if (cpu_pre_stress) {
      cpu_do_stress(cpu_scratchpad, cpu_pre_stress_iterations, cpu_pre_stress_pattern);
    }

    cpu_het_spin(&het_barriers[inst], HET_BARRIER_EXPECTED,
                 kernel_params->barrier_spin_limit);

#ifdef THREAD_0_IS_CPU
    // Thread 0: store x=1, fence, store y=1
    test_locations[x_addr].store(1, cuda::memory_order_relaxed);
#if defined(ALL_FENCE) || defined(FENCE_0) || defined(FENCE_01) || defined(FENCE_02)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#if defined(ACQ_REL) || defined(FENCE_1) || defined(FENCE_2) || defined(FENCE_12)
    test_locations[y_addr].store(1, cuda::memory_order_release);
#else
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: store y=2, fence, store z=1
    test_locations[y_addr].store(2, cuda::memory_order_relaxed);
#if defined(ALL_FENCE) || defined(FENCE_1) || defined(FENCE_01) || defined(FENCE_12)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#if defined(ACQ_REL) || defined(FENCE_0) || defined(FENCE_2) || defined(FENCE_02)
    test_locations[z_addr].store(1, cuda::memory_order_release);
#else
    test_locations[z_addr].store(1, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: r0 = load z, fence, r1 = load x
#if defined(ACQ_REL) || defined(FENCE_0) || defined(FENCE_1) || defined(FENCE_01)
    uint r0 = test_locations[z_addr].load(cuda::memory_order_acquire);
#else
    uint r0 = test_locations[z_addr].load(cuda::memory_order_relaxed);
#endif
#if defined(ALL_FENCE) || defined(FENCE_2) || defined(FENCE_02) || defined(FENCE_12)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
    uint r1 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
    read_results[inst].r1 = r1;
#endif
  }
}
