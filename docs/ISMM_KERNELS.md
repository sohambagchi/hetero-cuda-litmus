# ISMM Kernels

This document traces the full execution path for the ISMM experiment loop, from `ismm.sh` down to the exact CPU and GPU code that issues the memory operations.

It describes the current targeted ISMM path, not the older generic `runner.cu` or `run-full-matrix.sh` path.

## Overview

The ISMM flow is:

1. `ismm.sh` selects the next experiment name and invokes `ismm-runner`
2. `ismm_runner.cu` looks up that experiment in `kExperiments`
3. `ismm_runner.cu` builds a `RunConfig` with one role descriptor per litmus thread
4. GPU roles are executed by `gpu_role_kernel()` and `execute_gpu_role()`
5. CPU roles are executed by host threads running `cpu_role_worker()` and `execute_cpu_role()`
6. The actual memory operations are issued by:
   - CPU: `cpu_store_value()` and `cpu_load_value()`
   - GPU: `gpu_store_scoped()`, `gpu_load_scoped()`, `gpu_store_value()`, and `gpu_load_value()`
7. `classify_iteration()` checks whether the weak outcome occurred
8. `ismm.sh` appends the run summary into `full-matrix-results/ismm/results.csv`

## Shell Entry

File: `ismm.sh`

### Experiment list

`ismm.sh` defines the exact loop order in `EXPERIMENTS`.

Code:
- `ismm.sh:21`

### Runner compilation

If the targeted runner is missing or stale, `ismm.sh` compiles `ismm_runner.cu` into `target/ismm-runner-<backend>`.

Code:
- `ismm.sh:66`
- `ismm.sh:76`

### Infinite outer loop and resume behavior

`ismm.sh` computes `completed_runs` from the current number of data rows already present in `results.csv`, then uses modulo arithmetic to resume at the next experiment in the list.

Code:
- `ismm.sh:80`
- `ismm.sh:87`
- `ismm.sh:88`
- `ismm.sh:91`

### Per-experiment invocation

Each step in the loop runs:

```bash
"$RUNNER_BIN" -e "$experiment" -s "$STRESS_FILE" -t "$TEST_FILE"
```

Code:
- `ismm.sh:97`

### Result extraction and append

`ismm.sh` parses the runner stdout for:

- `Expectation:`
- `Time taken:`
- `Weak behavior rate:`
- `Total behaviors:`
- `Number of weak behaviors:`

and appends a row to `results.csv`.

Code:
- `ismm.sh:102`
- `ismm.sh:114`

## Experiment Selection

File: `ismm_runner.cu`

### Experiment table

The exact per-test role assignments are defined in the static experiment table `kExperiments`.

Code:
- `ismm_runner.cu:221`

This table chooses, for each litmus thread role:

- CPU or GPU domain
- GPU scope: system or device
- CPU store kind: none, relaxed, release
- CPU load kind: none, relaxed, acquire, rcsc, rcpc
- GPU store kind: none, relaxed, release
- GPU load kind: none, relaxed, acquire

Relevant declarations:
- `ismm_runner.cu:60`
- `ismm_runner.cu:65`
- `ismm_runner.cu:70`
- `ismm_runner.cu:75`
- `ismm_runner.cu:81`
- `ismm_runner.cu:89`
- `ismm_runner.cu:95`
- `ismm_runner.cu:101`
- `ismm_runner.cu:112`

### CLI parsing and experiment lookup

`main()` parses `-e`, `-s`, and `-t`, then resolves the experiment name via `find_experiment()`.

Code:
- `ismm_runner.cu:755`
- `ismm_runner.cu:763`
- `ismm_runner.cu:806`

Lookup helper:
- `ismm_runner.cu:456`

### Input parameter parsing

Stress parameters and test parameters are loaded from key-value files.

Code:
- `ismm_runner.cu:383`
- `ismm_runner.cu:408`
- `ismm_runner.cu:815`

In the current ISMM setup, the shell passes:

- stress file: `params-ismm.txt`
- test file: `params/2-loc.txt`

## Run Configuration Assembly

After experiment lookup, `main()` copies the selected experiment’s roles into `RunConfig`.

Code:
- `ismm_runner.cu:820`
- `ismm_runner.cu:828`

`RunConfig` carries:

- test kind: SB or IRIW
- number of litmus roles
- total instances
- address mapping parameters
- barrier spin limit
- per-role descriptors

Declaration:
- `ismm_runner.cu:120`

The total number of logical litmus instances is:

```text
totalInstances = workgroupSize * testingWorkgroups
```

Code:
- `ismm_runner.cu:823`

## Shared Memory and Scratch Allocation

The targeted runner allocates:

- shared test locations
- shared read results
- shared per-instance barriers
- GPU scratchpad and scratch locations when GPU roles exist

Code:
- `ismm_runner.cu:842`
- `ismm_runner.cu:851`
- `ismm_runner.cu:863`

The shared allocations use the repo’s common `het_malloc()` abstraction.

File: `memory_backends.h`

Code:
- `memory_backends.h:27`

This supports `MEM_HOSTALLOC`, `MEM_MANAGED`, and `MEM_MALLOC` backends.

## CPU Background Stress Infrastructure

The targeted runner also starts background CPU stress threads to perturb the memory system while the experiment loop runs.

Setup in `ismm_runner.cu`:
- `ismm_runner.cu:884`
- `ismm_runner.cu:892`

Worker function in `cpu_functions.h`:
- `cpu_functions.h:55`

## Per-Iteration Control Flow

The main experiment loop runs `stressParams.testIterations` times.

Code:
- `ismm_runner.cu:900`

At the start of each iteration it clears:

- test locations
- read results
- per-instance barriers

Code:
- `ismm_runner.cu:902`

The clears go through `het_memset()`.

File: `memory_backends.h`

Code:
- `memory_backends.h:63`

It then samples whether CPU and GPU pre-stress should run for this iteration.

Code:
- `ismm_runner.cu:906`

## GPU Execution Path

### GPU launch

If the chosen experiment includes GPU roles, `main()` launches `gpu_role_kernel<<<blocks, workgroupSize>>>`.

Code:
- `ismm_runner.cu:918`
- `ismm_runner.cu:924`

### Scratch-location preparation

Before launch, it fills `hScratchLocations` with random positions and copies them to the device.

Code:
- `ismm_runner.cu:695`
- `ismm_runner.cu:920`
- `ismm_runner.cu:921`

### Kernel dispatch model

`gpu_role_kernel()` treats the GPU work as a flat task space:

```text
taskIndex in [0, totalInstances * roleCount)
instance = taskIndex / roleCount
roleIndex = taskIndex % roleCount
```

Code:
- `ismm_runner.cu:617`
- `ismm_runner.cu:623`
- `ismm_runner.cu:629`

Each GPU task forwards into `execute_gpu_role()`.

Code:
- `ismm_runner.cu:631`

### GPU-side pre-stress and synchronization

Inside `execute_gpu_role()`:

1. optional pre-stress runs via `do_stress()`
2. the GPU participant joins a per-instance barrier with `het_spin()`

Code:
- `ismm_runner.cu:567`
- `ismm_runner.cu:579`
- `ismm_runner.cu:583`

Helper definitions:
- `functions.cu:21` for `do_stress()`
- `functions.cu:56` for `het_spin()`

### GPU address calculation

The GPU role computes:

- `xAddr = instance * memStride * 2`
- `yAddr = permute(instance, permuteLocation, totalInstances) * memStride * 2 + memOffset`

Code:
- `ismm_runner.cu:471`
- `ismm_runner.cu:475`
- `ismm_runner.cu:585`

On device, `y` uses the GPU helper `permute_id()`.

Helper definition:
- `functions.cu:4`

### GPU memory-operation helpers

The actual GPU loads and stores come from these helpers:

- `gpu_store_scoped()`
- `gpu_load_scoped()`
- `gpu_store_value()`
- `gpu_load_value()`

Code:
- `ismm_runner.cu:532`
- `ismm_runner.cu:543`
- `ismm_runner.cu:551`
- `ismm_runner.cu:559`

These helpers choose:

- system scope or device scope
- relaxed or release store
- relaxed or acquire load

The scope choice is made by dispatching to:

- `cuda::thread_scope_system`
- `cuda::thread_scope_device`

### GPU SB operations

Inside `execute_gpu_role()`, SB roles execute:

- role 0:
  - `store x = 1`
  - `r0 = load y`
- role 1:
  - `store y = 1`
  - `r1 = load x`

Code:
- `ismm_runner.cu:589`
- `ismm_runner.cu:591`
- `ismm_runner.cu:592`
- `ismm_runner.cu:595`
- `ismm_runner.cu:596`

### GPU IRIW operations

Inside `execute_gpu_role()`, IRIW roles execute:

- role 0:
  - `store x = 1`
- role 1:
  - `r0 = load x`
  - `r1 = load y`
- role 2:
  - `store y = 1`
- role 3:
  - `r2 = load y`
  - `r3 = load x`

Code:
- `ismm_runner.cu:602`
- `ismm_runner.cu:603`
- `ismm_runner.cu:605`
- `ismm_runner.cu:606`
- `ismm_runner.cu:609`
- `ismm_runner.cu:611`
- `ismm_runner.cu:612`

## CPU Execution Path

### CPU thread creation

If the chosen experiment includes CPU roles, `main()` creates a pool of host threads for each CPU role.

Code:
- `ismm_runner.cu:929`
- `ismm_runner.cu:931`
- `ismm_runner.cu:949`

Each host thread gets a `CpuWorkerContext` naming:

- the role it executes
- its worker index within that role
- the worker count for that role
- the shared memory objects
- a private scratchpad for pre-stress

Declaration:
- `ismm_runner.cu:635`

### CPU worker dispatch

Each host thread runs `cpu_role_worker()`, which walks instances in a strided pattern and calls `execute_cpu_role()`.

Code:
- `ismm_runner.cu:689`
- `ismm_runner.cu:691`

### CPU-side pre-stress and synchronization

Inside `execute_cpu_role()`:

1. optional pre-stress runs via `cpu_do_stress()`
2. the CPU participant joins the per-instance barrier via `cpu_het_spin()`

Code:
- `ismm_runner.cu:649`
- `ismm_runner.cu:655`
- `ismm_runner.cu:659`

Helper definitions:
- `cpu_functions.h:23` for `cpu_het_spin()`
- `cpu_functions.h:31` for `cpu_do_stress()`

### CPU address calculation

The CPU side uses the same logical address scheme as the GPU side:

- `xAddr = instance * memStride * 2`
- `yAddr = cpu_permute_id(instance, permuteLocation, totalInstances) * memStride * 2 + memOffset`

Code:
- `ismm_runner.cu:651`
- `ismm_runner.cu:652`

Permutation helper:
- `cpu_functions.h:12`

### CPU memory-operation helpers

The actual CPU loads and stores come from:

- `cpu_store_value()`
- `cpu_load_value()`

Code:
- `ismm_runner.cu:485`
- `ismm_runner.cu:502`

These helpers issue the current CPU-side memory operations:

- relaxed store via `atomic_ref<system>.store(..., memory_order_relaxed)`
- release store via `stlr` on AArch64
- relaxed load via `atomic_ref<system>.load(..., memory_order_relaxed)`
- acquire load via `atomic_ref<system>.load(..., memory_order_acquire)`
- rcsc acquire via `ldar` on AArch64
- rcpc acquire via `ldapr` on AArch64

Relevant lines:
- `ismm_runner.cu:487`
- `ismm_runner.cu:492`
- `ismm_runner.cu:505`
- `ismm_runner.cu:507`
- `ismm_runner.cu:511`
- `ismm_runner.cu:520`

### CPU SB operations

Inside `execute_cpu_role()`, SB roles execute:

- role 0:
  - `store x = 1`
  - `r0 = load y`
- role 1:
  - `store y = 1`
  - `r1 = load x`

Code:
- `ismm_runner.cu:661`
- `ismm_runner.cu:663`
- `ismm_runner.cu:664`
- `ismm_runner.cu:667`
- `ismm_runner.cu:668`

### CPU IRIW operations

Inside `execute_cpu_role()`, IRIW roles execute:

- role 0:
  - `store x = 1`
- role 1:
  - `r0 = load x`
  - `r1 = load y`
- role 2:
  - `store y = 1`
- role 3:
  - `r2 = load y`
  - `r3 = load x`

Code:
- `ismm_runner.cu:674`
- `ismm_runner.cu:675`
- `ismm_runner.cu:677`
- `ismm_runner.cu:678`
- `ismm_runner.cu:681`
- `ismm_runner.cu:683`
- `ismm_runner.cu:684`

## CPU/GPU Synchronization

The targeted runner uses a per-instance barrier stored in shared memory.

Type definition:
- `litmus_het.cuh:70`

GPU path:
- `functions.cu:56`

CPU path:
- `cpu_functions.h:23`

The expected participant count for the barrier is not hard-coded. It is passed dynamically as `config.roleCount`, so a pure CPU, pure GPU, or mixed experiment can all synchronize on the same instance.

GPU barrier call:
- `ismm_runner.cu:583`

CPU barrier call:
- `ismm_runner.cu:659`

## Result Classification

After all CPU threads join and the GPU kernel completes, `main()` calls `classify_iteration()`.

Code:
- `ismm_runner.cu:954`
- `ismm_runner.cu:957`
- `ismm_runner.cu:961`

Classifier:
- `ismm_runner.cu:710`

The classifier loads:

- final `x`
- final `y`
- recorded read results `r0`, `r1`, `r2`, `r3`

and checks the family-specific weak outcome.

SB weak check:
- `ismm_runner.cu:726`
- `ismm_runner.cu:729`

IRIW weak check:
- `ismm_runner.cu:737`
- `ismm_runner.cu:739`

## Runner Output Back To Shell

At the end of a run, `ismm_runner.cu` prints:

- experiment name
- expectation label
- elapsed time
- weak behavior rate
- total behaviors
- number of weak behaviors

Code:
- `ismm_runner.cu:969`
- `ismm_runner.cu:970`
- `ismm_runner.cu:971`
- `ismm_runner.cu:972`
- `ismm_runner.cu:975`
- `ismm_runner.cu:976`

`ismm.sh` parses those lines and appends the condensed row into `results.csv`.

Shell parser:
- `ismm.sh:102`
- `ismm.sh:106`
- `ismm.sh:114`

## Key Takeaway

The exact memory operations are not buried in the older litmus kernels under `kernels/`. For the current ISMM path they are issued directly from `ismm_runner.cu`:

- CPU operations: `execute_cpu_role()` -> `cpu_store_value()` / `cpu_load_value()`
- GPU operations: `gpu_role_kernel()` -> `execute_gpu_role()` -> `gpu_store_value()` / `gpu_load_value()`

That targeted runner is the code path behind `ismm.sh` and `full-matrix-results/ismm/results.csv`.
