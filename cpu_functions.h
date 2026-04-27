// cpu_functions.h — CPU-side functions for het litmus testing
#pragma once

#include <cuda/atomic>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>
#include "litmus_het.cuh"

// CPU-side permute_id (same formula as GPU)
inline uint cpu_permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

// CPU-side stripe_workgroup (same formula as GPU)
inline uint cpu_stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1) % testing_workgroups;
}

// CPU-side het barrier spin.
// Increments barrier, then spins until the count reaches expected_count.
inline void cpu_het_spin(het_barrier_t* barrier, uint expected_count, uint spin_limit) {
  barrier->fetch_add(1, cuda::memory_order_relaxed);
  for (uint i = 0; i < spin_limit; i++) {
    if (barrier->load(cuda::memory_order_acquire) >= expected_count) return;
  }
}

// CPU stress patterns (mirrors GPU do_stress)
inline void cpu_do_stress(volatile uint* scratchpad, uint iterations, uint pattern) {
  for (uint i = 0; i < iterations; i++) {
    if (pattern == 0) {
      scratchpad[0] = i;
      scratchpad[0] = i + 1;
    } else if (pattern == 1) {
      scratchpad[0] = i;
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
    } else if (pattern == 2) {
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
      scratchpad[0] = i;
    } else if (pattern == 3) {
      uint tmp = scratchpad[0];
      if (tmp > 100) break;
      uint tmp2 = scratchpad[0];
      if (tmp2 > 100) break;
    }
  }
}

// Background CPU stress thread function.
// Runs strided read/write across stress_array until *stop_flag is set.
inline void cpu_memory_stress_thread(volatile uint* stress_array, int array_size,
                                     volatile bool* stop_flag) {
  while (!*stop_flag) {
    for (int i = 1; i < array_size - 1; i++) {
      stress_array[i] = 1 + stress_array[i - 1] + stress_array[i + 1];
    }
  }
}
