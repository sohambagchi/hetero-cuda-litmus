# DESIGN-DECISIONS.md — Rationale for All Design Choices

This document records every design decision made during the planning phase, including the user's explicit choices and the technical rationale behind each.

---

## Decision 1: Instance Model — Multi-Instance

**Choice**: Multi-instance (thousands of test instances per kernel launch)

**Alternatives considered**: Single-instance (one test per launch, like SingleInstanceTestSuite)

**Rationale**: Single-instance testing has extremely low throughput — one weak behavior observation per kernel launch at best. cuda-litmus achieves ~1000× higher throughput by running thousands of instances in parallel. The same approach applies to het testing. More instances per launch means:
- Higher statistical power for detecting rare weak behaviors
- Better amortization of kernel launch overhead
- Ability to use the automated tuning system (which needs high throughput to compare configurations)

---

## Decision 2: Memory Sharing — Three Backends via Compile-Time #ifdef

**Choice**: Support `cudaHostAlloc` (mapped pinned), `cudaMallocManaged`, and `malloc` (unified SoC), selected via `MEM_HOSTALLOC` / `MEM_MANAGED` / `MEM_MALLOC` preprocessor flags.

**Alternatives considered**: 
- Runtime selection via function pointer / enum
- Single backend only

**Rationale**: Different hardware platforms require different memory sharing mechanisms:
- **Discrete GPU (PCIe)**: `cudaHostAlloc` with `cudaHostAllocMapped` is the standard way to share memory between CPU and GPU. The memory is pinned and mapped into both address spaces.
- **Any CUDA 6.0+ system**: `cudaMallocManaged` provides unified memory with automatic page migration. May have different performance characteristics.
- **Unified SoC (Grace-Hopper, Jetson)**: Plain `malloc` works because CPU and GPU share the same physical memory. No pinning or mapping needed. This is the highest-performance option on these platforms.

Compile-time selection avoids runtime overhead and allows the compiler to optimize for the specific backend. The memory allocation is abstracted in `memory_backends.h` so test code doesn't need to know which backend is active.

---

## Decision 3: Target Architecture — sm_90 (Hopper)

**Choice**: sm_90

**Rationale**: Grace-Hopper is the primary target platform for heterogeneous litmus testing because:
- It has a unified memory architecture (CPU and GPU share coherent memory)
- `thread_scope_system` operations are hardware-supported
- It's the latest NVIDIA architecture with the most interesting memory model behaviors to test
- The `malloc` backend is most natural on this platform

The code should also work on sm_80 (Ampere) and sm_87 (Jetson Orin) with the `cudaHostAlloc` or `cudaMallocManaged` backends, but sm_90 is the compilation target.

---

## Decision 4: Thread Counts — 2, 3, and 4 Thread Tests

**Choice**: Support all three test sizes from the cuda-litmus catalog.

**Rationale**: The 19 tests in the cuda-litmus catalog span 2-thread (MP, 2+2W, etc.), 3-thread (WRC, RWC, ISA2, etc.), and 4-thread (IRIW, etc.) configurations. Supporting all three is necessary for comprehensive coverage of the CUDA scoped memory model.

---

## Decision 5: Stress Mechanism — cuda-litmus Style

**Choice**: Full cuda-litmus stress infrastructure (18 tunable parameters, 4 stress patterns, pre-stress + mem-stress)

**Alternatives considered**: SingleInstanceTestSuite style (fixed 127-thread GPU stress + fixed CPU stress)

**Rationale**: cuda-litmus's parameterized stress system is far more powerful:
- 18 tunable parameters allow automated search over the stress space
- 4 distinct stress patterns (SS, SL, LS, LL) target different cache/pipeline effects
- Pre-stress (before test) and mem-stress (during test by non-testing threads) are independently controllable
- Workgroup shuffling randomizes SM assignment
- Stress assignment strategies (round-robin vs chunking) control contention patterns

The SingleInstanceTestSuite's fixed stress is a special case of this system.

---

## Decision 6: GPU Thread Topology — Full cuda-litmus Topology

**Choice**: Retain the full cuda-litmus topology system: `stripe_workgroup` for inter-block pairing, `permute_id` for intra-block permutation, TB_ macros for all configurations.

**Rationale**: The TB_ notation encodes the physical thread-block topology, which directly affects which cache levels are exercised:
- `TB_0_1` (inter-block): Threads in different SMs, must communicate through L2/DRAM
- `TB_01` (intra-block): Threads in the same SM, can communicate through L1/shared memory

For het tests, the TB_ config applies only to the GPU-side threads. CPU threads are always "external" to the GPU's thread-block hierarchy, so the TB_ config describes how GPU threads relate to each other when multiple GPU threads participate in a single test.

For test splits where only one thread is on the GPU (e.g., `HET_C0_G1`), the TB_ config is less meaningful but still affects the GPU thread's relationship to other GPU threads handling different instances.

---

## Decision 7: Het Thread Placement Notation

**Choice**: Flat enumeration per CPU/GPU split. Format: `HET_C<cpu_threads>_G<gpu_threads>` where each digit identifies a thread role from the litmus test.

**Examples for 2-thread test (MP)**:
- `HET_C0_G1` — Thread 0 on CPU, Thread 1 on GPU
- `HET_C1_G0` — Thread 1 on CPU, Thread 0 on GPU

**Examples for 3-thread test (WRC with threads 0, 1, 2)**:
- `HET_C0_G1_G2` — Thread 0 on CPU; Threads 1, 2 on GPU
- `HET_C0_C1_G2` — Threads 0, 1 on CPU; Thread 2 on GPU
- `HET_C0_G1_C2` — Threads 0, 2 on CPU; Thread 1 on GPU (non-contiguous)
- etc.

**Examples for 4-thread test (IRIW with threads 0, 1, 2, 3)**:
- `HET_C0_G1_G2_G3` — Thread 0 on CPU; Threads 1, 2, 3 on GPU
- `HET_C0_C1_G2_G3` — Threads 0, 1 on CPU; Threads 2, 3 on GPU
- etc.

**Rationale**: This notation is unambiguous and extensible. It clearly indicates which thread roles are CPU-side vs GPU-side. The compile-time `#ifdef` system uses these defines to conditionally compile CPU vs GPU code paths for each thread role.

---

## Decision 8: Variant Mechanism — Compile-Time #defines

**Choice**: Same as cuda-litmus: `-DSCOPE_SYSTEM -DACQUIRE`, `-DFENCE_SCOPE_DEVICE -DBOTH_FENCE`, etc.

**Alternatives considered**: Runtime selection, template parameters

**Rationale**: Compile-time defines allow the compiler to:
- Eliminate dead code branches (no runtime overhead)
- Inline atomic operations with known memory orders
- Optimize fence placement
- Generate different PTX for different scope/order combinations

This is critical for litmus testing where any overhead could mask weak behaviors.

---

## Decision 9: CPU-GPU Synchronization — Shared Atomic Spin-Wait Barrier

**Choice**: Per-instance barrier array using `cuda::atomic<uint, cuda::thread_scope_system>`.

```
barrier[i].fetch_add(1, relaxed);
while (barrier[i].load(relaxed) < num_threads_in_test) spin;
```

**Alternatives considered**:
- Global barrier (all instances sync together) — too coarse, serializes everything
- `cudaStreamSynchronize` + host event — too slow, kernel-launch granularity
- CUDA cooperative groups grid sync — only works within GPU

**Rationale**: The per-instance barrier is the minimum synchronization needed to ensure that all threads participating in a test instance have reached the test body before any of them execute test operations. Without this, the test would degenerate into purely sequential execution.

The barrier must use `thread_scope_system` because it spans CPU and GPU. Each instance has its own barrier element to avoid false sharing and contention between instances.

The spin limit (configurable via `barrierSpinLimit`) prevents deadlock if the GPU thread hasn't been scheduled yet.

---

## Decision 10: GPU Thread Participation — All Threads Participate

**Choice**: Every GPU thread in a testing workgroup handles a different test instance (cuda-litmus style).

**Alternatives considered**: Only thread 0 runs the test, rest do stress (SingleInstanceTestSuite style)

**Rationale**: This is the core multi-instance model. With 256 threads/block × 100 blocks = 25,600 instances per launch. Having only 1 test thread per block would reduce this to 100 instances — a 256× throughput reduction.

In the het model, each GPU thread handles one instance's GPU-side operations, while the corresponding CPU thread (from the pool) handles the same instance's CPU-side operations.

---

## Decision 11: CPU Thread Pool with Batches

**Choice**: M CPU threads (auto-detected), each processing K = ceil(total_instances / M) instances sequentially. Per-instance barriers synchronize CPU thread i with GPU thread for instance i.

**Rationale**: CPU threads are much more expensive to create than GPU threads, so we can't have one CPU thread per instance. Instead:
- The OS thread pool has M = `hardware_concurrency()` threads (typically 8-128)
- Total GPU instances = `workgroupSize × testing_workgroups` (typically thousands)
- Each CPU thread processes K = ceil(total/M) instances sequentially
- This means GPU threads will be waiting at the barrier while their CPU counterpart works through its batch

This creates a natural throughput bottleneck at the CPU side, which is acceptable because:
1. The barrier spin has a timeout, so if the CPU is slow, the GPU thread moves on
2. The batch processing means each CPU thread accesses contiguous memory, improving cache behavior
3. The tuning system can find the optimal `testing_workgroups` count that balances CPU/GPU throughput

---

## Decision 12: CPU Stress — Both Dedicated and Inline

**Choice**: Both dedicated CPU stress threads (background noise) AND CPU test threads do pre-stress before each instance.

**Rationale**:
- **Background stress threads**: Generate continuous memory bus contention (cache line bouncing, bus saturation). These run for the entire test duration, separate from test threads. Adapted from SingleInstanceTestSuite.
- **Inline pre-stress**: Each CPU test thread optionally executes a stress loop before hitting the per-instance barrier. This is analogous to GPU pre-stress and helps create timing variability.

Both mechanisms target different effects:
- Background stress creates sustained contention (affects cache line eviction, bus arbitration)
- Inline pre-stress creates per-instance timing variation (affects when threads reach the test body relative to each other)

---

## Decision 13: Build System — Shell Scripts + nvcc

**Choice**: Shell scripts (`tune.sh`) that invoke `nvcc` directly.

**Alternatives considered**: CMake, Makefile, Bazel

**Rationale**: 
- Direct `nvcc` invocation makes compile flags explicit and transparent
- Shell scripts are easy to modify for different platforms/configurations
- The tuning system needs to programmatically generate compile commands from the combinatorial space of (test × het_split × tb × scope × variant × fence × backend)
- No build system overhead — the project has a flat structure with few source files

---

## Decision 14: Automated Tuning — Full Tuning System

**Choice**: Direct adaptation of cuda-litmus's `tune.sh` with het-specific extensions.

**Rationale**: Finding weak memory behaviors is a needle-in-a-haystack problem. The stress parameter space has 18+ dimensions, and different configurations can differ by orders of magnitude in weak behavior detection rate. Automated random search is the proven approach:
1. Generate random stress configuration
2. Run all test variants with that configuration
3. Track which configurations produce the highest weak behavior rate
4. Run indefinitely, continuously improving

---

## Decision 15: Initial Test Set — Current Repo Inventory

**Choice**: Keep the current 19-test repo inventory as the supported suite and do not add doc-only `sb` or `lb` entries until they exist in the repository.

**Tests**:

| # | Test | Threads | Locs | Category |
|---|------|---------|------|----------|
| 1 | MP (Message Passing) | 2 | 2 | Store ordering |
| 2 | 2+2W | 2 | 2 | Write coherence |
| 3 | Read-Rel-Sys | 2 | 2 | Scope-specific |
| 4 | Read-Rel-Sys-And-CTA | 2 | 2 | Scope-specific |
| 5 | WRC | 3 | 2 | Causality |
| 6 | RWC | 3 | 2 | Causality |
| 7 | WWC | 3 | 2 | Causality |
| 8 | ISA2 | 3 | 3 | Causality |
| 9 | Z6.1 | 3 | 3 | Write serialization |
| 10 | Z6.3 | 3 | 3 | Write serialization |
| 11 | 3.2W | 3 | 3 | Write serialization |
| 12 | WRW+2W | 3 | 2 | Mixed |
| 13 | IRIW | 4 | 2 | Independent reads |
| 14 | IRIW-SC | 4 | 2 | Independent reads (SC) |
| 15 | IRIW-Extended | 4 | 4 | Independent reads |
| 16 | Counterexample | 4 | 3 | Scope mismatch |
| 17 | Paper Example | 4 | 4 | Paper-specific |
| 18 | Paper Example 1 | 4 | 4 | Paper-specific |
| 19 | Paper Example 2 | 4 | 4 | Paper-specific |

**Rationale**: These tests comprehensively cover the CUDA scoped memory model, including:
- Classic litmus tests (MP, SB, LB, 2+2W, IRIW)
- Multi-copy atomicity tests (WRC, RWC, WWC, ISA2)
- Write serialization tests (Z6.1, Z6.3, 3.2W, WRW+2W)
- Scope-specific tests (counterexample, read-rel-sys variants)

Adapting all of them for het execution allows comprehensive cross-device memory model testing.

---

## Decision 16: Atomic Types for Het Tests

**Choice**: Shared test memory uses a `het_atomic_uint` wrapper over `uint` storage, with GPU accesses going through `cuda::atomic_ref<..., TEST_SCOPE>` and CPU accesses staying system-scope.

**Rationale**: Cross-device sharing still requires system-visible backing storage, but the test suite also needs to exercise narrower GPU scopes. Wrapping a shared `uint` with `cuda::atomic_ref` allows the GPU side to test `SCOPE_DEVICE` and `SCOPE_BLOCK` while keeping CPU-side accesses valid at system scope.

The `SCOPE_SYSTEM` / `SCOPE_DEVICE` / `SCOPE_BLOCK` compile flags control the GPU-side test-operation scope. CPU-side operations remain system-scope because CPU execution is outside the GPU block/device hierarchy.

This is a key distinction:
- **Shared backing storage** — ensures CPU and GPU can both access the same location
- **GPU operation scope** (configurable) — what the test is actually testing
- **CPU operation scope** (system) — the fixed cross-device endpoint

---

## Decision 17: CPU Atomic Operations

**Choice**: CPU threads use `cuda::atomic<uint, cuda::thread_scope_system>` (libcu++ types) for test operations, matching the GPU side.

**Alternatives considered**: `std::atomic<uint>` with `memory_order_*`

**Rationale**: Using the same `cuda::atomic` type on both CPU and GPU:
- Ensures identical memory ordering semantics (libcu++ guarantees cross-CPU/GPU ordering for system scope)
- Avoids subtle ABI mismatches between `std::atomic` and `cuda::atomic`
- libcu++ is specifically designed for heterogeneous CPU-GPU atomics
- Simplifies the code — same type on both sides

libcu++ supports host-side compilation of `cuda::atomic` operations via the `-rdc=true` flag and appropriate includes.

---

## Decision 18: Result Collection — GPU-side check_results Kernel

**Choice**: Same as cuda-litmus: a separate `check_results` GPU kernel classifies outcomes.

**Rationale**: The read_results array is in shared memory (accessible by both CPU and GPU), so the classification kernel can run on the GPU where it benefits from massive parallelism. Each GPU thread classifies one instance independently.

An alternative would be host-side classification, but this would be slower for thousands of instances and would require copying data back to host memory (unless using the `malloc` backend where it's already there).

The `check_results` kernel uses the same pattern as cuda-litmus: check each register combination and increment the appropriate result counter atomically.
