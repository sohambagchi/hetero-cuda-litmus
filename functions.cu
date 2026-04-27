// functions.cu — GPU device functions
// Identical to cuda-litmus, plus het_spin for per-instance CPU-GPU barriers.

__device__ uint permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

__device__ uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1) % testing_workgroups;
}

__device__ void spin(cuda::atomic<uint, cuda::thread_scope_device>* barrier, uint limit) {
  int i = 0;
  uint val = barrier->fetch_add(1, cuda::memory_order_relaxed);
  while (i < 1024 && val < limit) {
    val = barrier->load(cuda::memory_order_relaxed);
    i++;
  }
}

__device__ void do_stress(uint* scratchpad, uint* scratch_locations, uint iterations, uint pattern) {
  for (uint i = 0; i < iterations; i++) {
    if (pattern == 0) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      scratchpad[scratch_locations[blockIdx.x]] = i + 1;
    }
    else if (pattern == 1) {
      scratchpad[scratch_locations[blockIdx.x]] = i;
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
    }
    else if (pattern == 2) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
      scratchpad[scratch_locations[blockIdx.x]] = i;
    }
    else if (pattern == 3) {
      uint tmp1 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp1 > 100) {
        break;
      }
      uint tmp2 = scratchpad[scratch_locations[blockIdx.x]];
      if (tmp2 > 100) {
        break;
      }
    }
  }
}

// GPU thread waits at per-instance het barrier.
// Increments barrier, then spins until the count reaches expected_count.
__device__ void het_spin(het_barrier_t* barrier, uint expected_count, uint spin_limit) {
  barrier->fetch_add(1, cuda::memory_order_relaxed);
  for (uint i = 0; i < spin_limit; i++) {
    if (barrier->load(cuda::memory_order_acquire) >= expected_count) return;
  }
}
