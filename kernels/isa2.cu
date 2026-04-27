// kernels/isa2.cu — ISA2 (het)
// Thread 0: store x=1, FENCE_0(), store y=1
// Thread 1: r0 = load y, FENCE_1(), store z=1
// Thread 2: r1 = load z, FENCE_2(), r2 = load x
// Weak behavior: r0==1 && r1==1 && r2==0

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
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2()
#elif defined(RELEASE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2()
#elif defined(RELAXED)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2()
#elif defined(ALL_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_0_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1()
    #define FENCE_2()
#elif defined(THREAD_0_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1()
    #define FENCE_2()
#elif defined(THREAD_1_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0()
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2()
#elif defined(THREAD_2_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_2_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_0_1_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_acquire;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2()
#elif defined(THREAD_0_2_FENCE_ACQ)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_acquire;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_0_2_FENCE_REL)
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_release;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_1()
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#elif defined(THREAD_1_2_FENCE)
    cuda::memory_order thread_0_store = cuda::memory_order_release;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
    #define FENCE_2() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, FENCE_SCOPE);
#else
    cuda::memory_order thread_0_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_load = cuda::memory_order_relaxed;
    cuda::memory_order thread_1_store = cuda::memory_order_relaxed;
    cuda::memory_order thread_2_load = cuda::memory_order_relaxed;
    #define FENCE_0()
    #define FENCE_1()
    #define FENCE_2()
#endif

    DEFINE_IDS();
    THREE_THREAD_THREE_MEM_LOCATIONS();

    if (kernel_params->pre_stress) {
      do_stress(scratchpad, scratch_locations, kernel_params->pre_stress_iterations, kernel_params->pre_stress_pattern);
    }
    if (kernel_params->barrier) {
      spin(gpu_barrier, blockDim.x * kernel_params->testing_workgroups);
    }

    if (true) {

#ifdef THREAD_0_IS_GPU
      het_spin(&het_barriers[wg_offset + id_0], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 0: store x=1, fence, store y=1
      test_locations[x_0].store(1, cuda::memory_order_relaxed);
      FENCE_0()
      test_locations[y_0].store(1, thread_0_store);
#endif

#ifdef THREAD_1_IS_GPU
      het_spin(&het_barriers[wg_offset + id_1], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 1: r0 = load y, fence, store z=1
      uint r0 = test_locations[y_1].load(thread_1_load);
      FENCE_1()
      test_locations[z_1].store(1, thread_1_store);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_1].r0 = r0;
#endif

#ifdef THREAD_2_IS_GPU
      het_spin(&het_barriers[wg_offset + id_2], HET_BARRIER_EXPECTED,
               kernel_params->barrier_spin_limit);
      // Thread 2: r1 = load z, fence, r2 = load x
      uint r1 = test_locations[z_2].load(thread_2_load);
      FENCE_2()
      uint r2 = test_locations[x_2].load(cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
      read_results[wg_offset + id_2].r1 = r1;
      read_results[wg_offset + id_2].r2 = r2;
#endif
    }
  }
  else if (kernel_params->mem_stress) {
    do_stress(scratchpad, scratch_locations, kernel_params->mem_stress_iterations, kernel_params->pre_stress_iterations);
  }
}

__global__ void check_results(
  het_atomic_uint* test_locations,
  ReadResults* read_results,
  TestResults* test_results,
  KernelParams* kernel_params,
  bool* weak) {
  uint id_0 = blockIdx.x * blockDim.x + threadIdx.x;
  uint r0 = read_results[id_0].r0;
  uint r1 = read_results[id_0].r1;
  uint r2 = read_results[id_0].r2;
  uint x = test_locations[id_0 * kernel_params->mem_stride * 3];

  if (x == 0) {
    test_results->na.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 1) {
    test_results->res0.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 0) {
    test_results->res1.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 0 && r2 == 1) {
    test_results->res2.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 0) {
    test_results->res3.fetch_add(1);
  }
  else if (r0 == 0 && r1 == 1 && r2 == 1) {
    test_results->res4.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 0) {
    test_results->res5.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 0 && r2 == 1) {
    test_results->res6.fetch_add(1);
  }
  else if (r0 == 1 && r1 == 1 && r2 == 0) {
    test_results->weak.fetch_add(1);
    weak[id_0] = true;
  }
  else {
    test_results->other.fetch_add(1);
  }
}

int host_check_results(TestResults* results, bool print) {
  if (print) {
    std::cout << "r0=0, r1=1, r2=1 (seq): " << results->res0 << "\n";
    std::cout << "r0=0, r1=0, r2=0 (seq): " << results->res1 << "\n";
    std::cout << "r0=0, r1=0, r2=1 (seq): " << results->res2 << "\n";
    std::cout << "r0=0, r1=1, r2=0: " << results->res3 << "\n";
    std::cout << "r0=0, r1=1, r2=1: " << results->res4 << "\n";
    std::cout << "r0=1, r1=0, r2=0 (seq): " << results->res5 << "\n";
    std::cout << "r0=1, r1=0, r2=1 (interleaved): " << results->res6 << "\n";
    std::cout << "r0=1, r1=1, r2=0 (weak): " << results->weak << "\n";
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
    // 3 memory locations: x, y, z
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
#if defined(ALL_FENCE) || defined(THREAD_0_FENCE_ACQ) || defined(THREAD_0_FENCE_REL) || defined(THREAD_0_1_FENCE) || defined(THREAD_0_2_FENCE_ACQ) || defined(THREAD_0_2_FENCE_REL)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#if defined(ACQUIRE) || defined(RELEASE) || defined(THREAD_1_FENCE) || defined(THREAD_2_FENCE_ACQ) || defined(THREAD_2_FENCE_REL) || defined(THREAD_1_2_FENCE)
    test_locations[y_addr].store(1, cuda::memory_order_release);
#else
    test_locations[y_addr].store(1, cuda::memory_order_relaxed);
#endif
#endif

#ifdef THREAD_1_IS_CPU
    // Thread 1: r0 = load y, fence, store z=1
#if defined(ACQUIRE) || defined(THREAD_0_FENCE_ACQ) || defined(THREAD_2_FENCE_ACQ) || defined(THREAD_0_2_FENCE_ACQ)
    uint r0 = test_locations[y_addr].load(cuda::memory_order_acquire);
#else
    uint r0 = test_locations[y_addr].load(cuda::memory_order_relaxed);
#endif
#if defined(ALL_FENCE) || defined(THREAD_1_FENCE) || defined(THREAD_0_1_FENCE) || defined(THREAD_1_2_FENCE)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
#if defined(RELEASE) || defined(THREAD_0_FENCE_REL) || defined(THREAD_2_FENCE_REL) || defined(THREAD_0_2_FENCE_REL)
    test_locations[z_addr].store(1, cuda::memory_order_release);
#else
    test_locations[z_addr].store(1, cuda::memory_order_relaxed);
#endif
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r0 = r0;
#endif

#ifdef THREAD_2_IS_CPU
    // Thread 2: r1 = load z, fence, r2 = load x
#if defined(ACQUIRE) || defined(RELEASE) || defined(THREAD_0_FENCE_ACQ) || defined(THREAD_0_FENCE_REL) || defined(THREAD_0_1_FENCE)
    uint r1 = test_locations[z_addr].load(cuda::memory_order_acquire);
#else
    uint r1 = test_locations[z_addr].load(cuda::memory_order_relaxed);
#endif
#if defined(ALL_FENCE) || defined(THREAD_2_FENCE_ACQ) || defined(THREAD_2_FENCE_REL) || defined(THREAD_0_2_FENCE_ACQ) || defined(THREAD_0_2_FENCE_REL) || defined(THREAD_1_2_FENCE)
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
#endif
    uint r2 = test_locations[x_addr].load(cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    read_results[inst].r1 = r1;
    read_results[inst].r2 = r2;
#endif
  }
}
