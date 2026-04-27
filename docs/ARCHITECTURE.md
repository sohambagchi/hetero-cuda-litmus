# ARCHITECTURE.md — NVIDIA Heterogeneous Litmus Testing Framework

## Overview

This project is a **multi-instance, heterogeneous CPU-GPU litmus testing framework** targeting NVIDIA GPUs (sm_90 / Hopper). It combines the multi-instance, parameterized stress infrastructure of `cuda-litmus` (GPU-only) with the CPU-GPU shared-memory testing model of `SingleInstanceTestSuite`.

The key innovation: **thousands of test instances per kernel launch**, where each instance involves CPU threads and GPU threads cooperating on shared memory, enabling high-throughput detection of cross-device memory consistency violations.

---

## Execution Model

### Conceptual Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Host (runner.cu)                             │
│                                                                     │
│  for each iteration:                                                │
│    1. Zero shared memory, set params                                │
│    2. Launch GPU kernel (N_total blocks × blockDim threads)         │
│    3. Launch CPU thread pool (M threads, K instances each)          │
│    4. Per-instance barrier synchronizes CPU thread ↔ GPU thread     │
│    5. Both sides execute their test operations                      │
│    6. Join CPU threads, synchronize GPU                             │
│    7. Launch check_results kernel OR host-side result check         │
│    8. Accumulate results                                            │
│                                                                     │
│  After all iterations: print summary, save to file                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Instance Mapping

Each test instance `i` (where `i ∈ [0, total_instances)`) is assigned:
- **GPU side**: One or more GPU threads (determined by TB_ topology and stripe/permute)
- **CPU side**: One or more CPU threads from the thread pool

For a 2-thread test like Message Passing with split `HET_C0_G1`:
- Thread 0 (producer) runs on a CPU thread
- Thread 1 (consumer) runs on a GPU thread

The GPU thread for instance `i` is the same physical thread that would handle instance `i` in the original cuda-litmus framework. The CPU thread for instance `i` is determined by the thread pool batch assignment: CPU thread `j` handles instances `[j*K, (j+1)*K)`.

### Per-Instance Barrier

```
Shared barrier array: cuda::atomic<uint, cuda::thread_scope_system> barrier[total_instances]

CPU thread for instance i:          GPU thread for instance i:
  barrier[i].fetch_add(1, relaxed)    barrier[i].fetch_add(1, relaxed)
  while (barrier[i].load() < 2)      while (barrier[i].load() < 2)
    spin;                               spin;
  // execute test ops                 // execute test ops
```

Both sides must use `thread_scope_system` for the barrier since it spans CPU and GPU.

### Thread Pool Model

```
total_instances = workgroupSize × testing_workgroups  (same as cuda-litmus)
M = hardware_concurrency()   (auto-detected CPU thread count)
K = ceil(total_instances / M) (instances per CPU thread)

CPU thread j processes instances: j*K, j*K+1, ..., min((j+1)*K - 1, total_instances - 1)
```

Each CPU thread, for each of its assigned instances:
1. Optionally does pre-stress (cache/memory bus noise)
2. Hits the per-instance barrier
3. Executes its test operations (stores/loads on shared memory)
4. Records read results into the shared read_results array

---

## Memory Sharing Backends

Three backends, selected at compile time via `#ifdef`:

| Backend | Flag | Allocation | GPU Access | Use Case |
|---------|------|-----------|------------|----------|
| Pinned mapped | `MEM_HOSTALLOC` | `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer` | Via mapped pointer | Discrete GPUs, PCIe systems |
| Managed | `MEM_MANAGED` | `cudaMallocManaged(...)` | Direct (unified) | Any CUDA ≥ 6.0 system |
| Plain malloc | `MEM_MALLOC` | `malloc(...)` | Direct (unified SoC) | Grace-Hopper, Jetson (sm_87/sm_90) |

All test locations, read results, barriers, and instance metadata use the selected backend. The shared test-location type is `het_atomic_uint`, a wrapper over shared `uint` storage that uses `cuda::atomic_ref` so GPU-side operations honor `TEST_SCOPE` while CPU-side operations stay system-scope.

---

## File Structure

```
NVIDIA-Het-Litmus/
├── docs/
│   ├── ARCHITECTURE.md          (this file)
│   ├── DESIGN-DECISIONS.md      (rationale for all design choices)
│   ├── IMPLEMENTATION-GUIDE.md  (per-file specs, data structures, code patterns)
│   └── TEST-CATALOG.md          (all litmus tests with het splits)
│
├── litmus_het.cuh               Core header: types, macros, data structures
├── functions.cu                 Device functions: permute_id, stripe_workgroup, spin, do_stress
├── cpu_functions.h              CPU functions: cpu_do_stress, cpu_het_spin
├── runner.cu                    Host-side runner: alloc, launch, collect
├── memory_backends.h            Memory allocation abstraction (3 backends)
│
├── kernels/                     Test kernel files (one per litmus test)
│   ├── mp.cu                    Message Passing (2-thread)
│   ├── 2+2w.cu                  Two Plus Two Writes (2-thread)
│   ├── wrc.cu                   Write-Read Causality (3-thread)
│   ├── rwc.cu                   Read-Write Causality (3-thread)
│   ├── wwc.cu                   Write-Write Causality (3-thread)
│   ├── isa2.cu                  ISA2 (3-thread)
│   ├── z6-1.cu                  Z6.1 (3-thread)
│   ├── z6-3.cu                  Z6.3 (3-thread)
│   ├── 3.2w.cu                  3.2W (3-thread)
│   ├── wrw+2w.cu                WRW+2W (3-thread)
│   ├── iriw.cu                  IRIW (4-thread)
│   ├── iriw-sc.cu               IRIW-SC (4-thread)
│   ├── counterexample.cu        Counterexample (4-thread, scope test)
│   ├── read-rel-sys.cu          Read-Rel-Sys (2-thread)
│   ├── read-rel-sys-and-cta.cu  Read-Rel-Sys-And-CTA (2-thread)
│   ├── paper-example.cu         Paper Example (4-thread)
│   ├── paper-example1.cu        Paper Example 1 (4-thread)
│   └── paper-example2.cu        Paper Example 2 (4-thread)
│
├── params/                      Test parameter files
│   ├── 2-loc.txt                numMemLocations=2, permuteLocation=1031
│   ├── 3-loc.txt                numMemLocations=3, permuteLocation=1031
│   └── 4-loc.txt                numMemLocations=4, permuteLocation=1031
│
├── tuning-files/                Tuning configuration files (one per test)
│   ├── mp.txt
│   ├── 2+2w.txt
│   ├── wrc.txt
│   ├── rwc.txt
│   ├── wwc.txt
│   ├── isa2.txt
│   ├── z6-1.txt
│   ├── z6-3.txt
│   ├── 3.2w.txt
│   ├── wrw+2w.txt
│   ├── iriw.txt
│   ├── iriw-sc.txt
│   ├── iriw-extended.txt
│   ├── counterexample.txt
│   ├── paper-example.txt
│   ├── paper-example1.txt
│   └── paper-example2.txt
│
├── all-tests.txt                Current tuning inventory
├── params-smoke.txt             Small verification parameter set
├── run-full-matrix.sh           Offline full-matrix compile/run helper
├── tune.sh                      Automated tuning script
├── analyze.py                   Result analysis script
│
├── target/                      Compiled binaries (generated)
└── results/                     Tuning results (generated)
```

---

## Compilation Model

Each test binary is compiled with a specific combination of:
1. **HET split** — e.g., `-DHET_C0_G1` (which threads are CPU vs GPU)
2. **TB topology** — e.g., `-DTB_0_1` (thread-block assignment for GPU threads)
3. **Scope** — e.g., `-DSCOPE_SYSTEM` (atomic scope for test operations)
4. **Variant** — e.g., `-DACQUIRE` (memory ordering for test operations)
5. **Fence scope** — e.g., `-DFENCE_SCOPE_DEVICE` (scope for explicit fences)
6. **Memory backend** — e.g., `-DMEM_HOSTALLOC`

```bash
/usr/local/cuda-12.4/bin/nvcc -DTB_0_1 -DHET_C0_G1 -DSCOPE_SYSTEM -DACQ_REL -DMEM_MALLOC \
     -I. -rdc=true -arch sm_90 \
     runner.cu kernels/mp.cu \
     -o target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-malloc-runner
```

The het split defines (`HET_C0_G1`, etc.) control which thread roles are executed on the CPU vs GPU. This is the primary new compile-time dimension compared to cuda-litmus.

---

## Stress Mechanisms

### GPU Stress (from cuda-litmus)
- **Pre-stress**: Before the test body, testing workgroups optionally execute `do_stress()` — store-store, store-load, load-store, or load-load patterns on scratchpad memory.
- **Mem-stress**: Non-testing workgroups (those beyond `testing_workgroups`) execute stress patterns instead of the test, generating memory bus contention.
- **Barrier**: Optional `spin()` barrier after pre-stress to synchronize testing workgroups.
- **Workgroup shuffling**: `shuffled_workgroups[]` array randomizes block-to-SM mapping each iteration.

### CPU Stress (from SingleInstanceTestSuite, extended)
- **Background stress threads**: Dedicated `std::thread`s that continuously read/write a shared array in strided patterns, generating cache/bus contention. Run for the entire test duration.
- **CPU pre-stress**: Each CPU test thread optionally executes a stress loop before hitting the barrier, similar to GPU pre-stress but using CPU cache line operations.
- **CPU stress patterns**: Same 4 patterns (store-store, store-load, load-store, load-load) adapted for CPU.

### Combined Stress Parameters (18 tunable + het-specific)

All cuda-litmus stress parameters are preserved:
- `testIterations`, `testingWorkgroups`, `maxWorkgroups`, `workgroupSize`
- `shufflePct`, `barrierPct`
- `stressLineSize`, `stressTargetLines`, `scratchMemorySize`
- `memStride`, `memStressPct`, `memStressIterations`, `memStressPattern`
- `preStressPct`, `preStressIterations`, `preStressPattern`
- `stressAssignmentStrategy`, `permuteThread`

New het-specific parameters:
- `cpuStressThreads` — number of background CPU stress threads (default: hardware_concurrency / 2)
- `cpuPreStressPct` — percentage of iterations where CPU threads do pre-stress
- `cpuPreStressIterations` — iterations for CPU pre-stress loop
- `cpuPreStressPattern` — stress pattern for CPU pre-stress (0-3)
- `barrierSpinLimit` — max spin iterations for per-instance barrier timeout

---

## Result Classification

Same model as cuda-litmus: a separate `check_results` kernel (or host function) classifies each instance's outcome into one of 16 result buckets + `weak` + `na` + `other`.

- **`weak`**: The specific relaxed memory behavior being tested (e.g., for MP: `r0=1, r1=0`)
- **`na`**: Instance was marked not-applicable by the test's result checker
- **`res0..res15`**: All other valid outcome combinations
- **`other`**: Unexpected results (should always be 0)

For het tests, the `check_results` kernel runs on the GPU and reads from the shared `read_results` array (which was populated by both CPU and GPU threads).

---

## Tuning System

The tuning system is a direct adaptation of cuda-litmus's `tune.sh`:

1. **Compile phase**: Build all (test × het_split × tb × scope × variant) combinations
2. **Tuning loop**: Infinite loop generating random stress parameter configurations
3. **Execution**: Run each binary with the random config for `testIterations` iterations
4. **Tracking**: If weak behaviors are found, save the config. If the rate exceeds the previous best, update.
5. **Output**: `results/<test-config>/params.txt` + `results/<test-config>/rate`

The tuning files specify the combinatorial space per test:
```
<test_name> <param_file>
<TB configs>
<HET split configs>
<SCOPE options>
<non-fence variants>
[<FENCE_SCOPE options>]
[<fence variants>]
```

For exhaustive offline verification, use `run-full-matrix.sh` rather than the interactive `tune.sh` loop. It walks the same tuning-file space, compiles each binary, optionally runs each one, and writes logs plus CSV summaries under `full-matrix-results/`.

---

## Key Differences from cuda-litmus

| Aspect | cuda-litmus | NVIDIA-Het-Litmus |
|--------|-------------|-------------------|
| Threads | All GPU | CPU + GPU (heterogeneous) |
| Memory | `cudaMalloc` (device-only) | Shared memory (3 backends) |
| Atomic scope | device/block | GPU operations honor `SCOPE_*`; CPU operations remain system-scope |
| Barrier | Device-scope atomic counter | **System-scope** atomic counter array |
| Instances per barrier | One global barrier for all testing threads | **Per-instance** barrier (array) |
| CPU involvement | Host-side runner only | CPU threads execute test operations |
| Thread assignment | `DEFINE_IDS()` macros only | `DEFINE_IDS()` for GPU + thread pool for CPU |
| New compile dimension | — | HET split (`HET_C0_G1`, etc.) |
| Target arch | sm_80 | sm_90 |

---

## Key Differences from SingleInstanceTestSuite

| Aspect | SingleInstanceTestSuite | NVIDIA-Het-Litmus |
|--------|------------------------|-------------------|
| Instances per launch | 1 | Thousands |
| GPU config | 1 block, 128 threads (1 test + 127 stress) | N blocks × W threads (all participate) |
| Parameterization | Hardcoded | 18+ tunable stress params |
| Tests | 6 basic (MP, SB, LB, Read, Store, 2+2W) | 19 tests in the current repo inventory |
| Variants | Per-file source variants | Compile-time `-D` flags |
| Tuning | Manual | Automated random search |
| Result collection | Per-iteration host check | Batch result classification kernel |
