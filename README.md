# NVIDIA-Het-Litmus

Heterogeneous CPU-GPU litmus test framework for NVIDIA GPUs. This README summarizes the build targets, helper scripts, runner flags, and the common ways to compile and run the tests in this repository.

## Requirements

- CUDA toolkit with `nvcc` available. The project defaults to `/usr/local/cuda-12.4/bin/nvcc`.
- A GPU architecture supported by your machine. The default build target is `sm_90`.
- Run commands from `NVIDIA-Het-Litmus/`.

## Repository Inputs And Outputs

- `all-tests.txt`: default inventory of tuning files to compile or sweep.
- `tuning-files/*.txt`: per-test compile matrix definitions.
- `params/*.txt`: per-test parameter files passed to the runner with `-t`.
- `params-smoke.txt`: small stress configuration useful for quick validation.
- `target/`: compiled runners.
- `results/`: best tuning results recorded by `tune.sh`.
- `full-matrix-results/`: compile/run logs and CSV summaries recorded by `run-full-matrix.sh`.

## Makefile Targets

`make` defaults to `compile-only`.

| Target | What it does |
| --- | --- |
| `make compile-only` | Compile all binaries described by `TESTS` using `MEM_BACKEND`.
| `make compile-managed` | Compile with `MEM_BACKEND=MANAGED`.
| `make compile-malloc` | Compile with `MEM_BACKEND=MALLOC`.
| `make tune` | Compile, then start the infinite tuning loop.
| `make tune-no-compile` | Start the tuning loop without recompiling first.
| `make clean` | Remove `target/`, `results/`, and `params.txt`.
| `make help` | Print the built-in usage summary.

### Makefile Variables

These can be overridden on the command line.

| Variable | Meaning | Default |
| --- | --- | --- |
| `TESTS` | File listing tuning files to compile or tune | `all-tests.txt` |
| `MEM_BACKEND` | Shared-memory backend: `HOSTALLOC`, `MANAGED`, or `MALLOC` | `HOSTALLOC` |
| `NVCC` | CUDA compiler path | `/usr/local/cuda-12.4/bin/nvcc` |
| `ARCH` | CUDA architecture passed to `nvcc -arch` | `sm_90` |
| `HET_DEBUG` | Set to `1` to compile with extra runner diagnostics | `0` |

### Parallel Compilation

- `make -jN compile-only` compiles up to `N` binaries in parallel.
- Example: `make -j8 compile-only MEM_BACKEND=MANAGED`

## Scripts And Flags

### `tune.sh`

Usage:

```bash
./tune.sh <tuning-list-file> [--no-compile] [--mem-backend <HOSTALLOC|MANAGED|MALLOC>]
```

Purpose:

- Optionally compiles all combinations from the tuning list.
- Generates random stress parameters into `params.txt`.
- Re-runs the compiled binaries forever until interrupted.
- Saves the best observed weak-behavior rate for each configuration under `results/`.

Flags:

| Flag | Meaning |
| --- | --- |
| `<tuning-list-file>` | Required file containing one tuning-file path per line, such as `all-tests.txt` |
| `--no-compile` | Skip compilation and only run the tuning loop |
| `--mem-backend <...>` | Select `HOSTALLOC`, `MANAGED`, or `MALLOC` |

Environment variables used by `tune.sh`:

- `NVCC`: compiler path
- `ARCH`: CUDA architecture
- `HET_DEBUG`: `0` or `1`

### `run-full-matrix.sh`

Usage:

```bash
./run-full-matrix.sh [options]
```

Purpose:

- Builds the full compile/run matrix described by the tuning files.
- Can compile only, run only, or do both.
- Writes logs plus CSV summaries for every configuration.

Flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--mode <compile-and-run|compile-only|run-only>` | Select pipeline mode | `compile-and-run` |
| `--tests <family[,family...]|family family...>` | Restrict the matrix to one or more test families | `all` |
| `--mem-backend <HOSTALLOC|MANAGED|MALLOC>` | Memory backend | `MALLOC` |
| `--nvcc <path>` | CUDA compiler path | `/usr/local/cuda-12.4/bin/nvcc` |
| `--arch <arch>` | GPU architecture | `sm_90` |
| `--stress <file>` | Stress parameter file passed to each runner with `-s` | `params-smoke.txt` |
| `-j`, `--jobs <count>` | Parallel compile workers | `1` |
| `--out-dir <dir>` | Output directory for logs and CSV summaries | `full-matrix-results/<timestamp>` |
| `--resume <dir>` | Resume an interrupted matrix run in an existing output directory | unset |
| `--target-dir <dir>` | Directory containing compiled runners | `target` |
| `--tests-file <file>` | Tuning-list file used to build the matrix | `all-tests.txt` |
| `--filter-test <name>` | Restrict the matrix to one test name | unset |
| `--help` | Print usage |  |

Notes:

- `-j/--jobs` affects compilation only.
- Execution stays serial so `run.log` remains ordered.
- `--resume` requires the same matrix configuration and reuses the original `matrix.tsv` plus status files to skip completed entries.
- `HET_DEBUG` is read from the environment and must be `0` or `1`.

### `analyze.py`

Usage:

```bash
python3 analyze.py [results_dir] [--csv FILE] [--log FILE]
```

Purpose:

- Summarizes `results/` produced by `tune.sh`.
- Can also analyze raw `tune.sh` console output saved to a log file.

Flags:

| Flag | Meaning |
| --- | --- |
| `results_dir` | Results directory to inspect, default `results` |
| `--csv FILE` | Export the analyzed summary to CSV |
| `--log FILE` | Analyze raw `tune.sh` log output instead of `results/` |

## Running Litmus Tests

There are three common ways to run tests.

## Fast Path

This path is only for the `MALLOC` backend.

Use it when you want the shortest commands for compile, direct execution, matrix sweeps, or tuning with `MEM_MALLOC`.

### Compile Everything With MALLOC

```bash
make compile-malloc
```

Parallel version:

```bash
make -j8 compile-malloc
```

### Run One MALLOC Binary Directly

```bash
./target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-malloc-runner \
  -s params-smoke.txt \
  -t params/2-loc.txt
```

### Run A MALLOC Full-Matrix Smoke Sweep

```bash
./run-full-matrix.sh --mode compile-and-run --mem-backend MALLOC --stress params-smoke.txt -j 4
```

Restrict to a single test:

```bash
./run-full-matrix.sh --mode compile-and-run --mem-backend MALLOC --stress params-smoke.txt --filter-test mp -j 4
```

Restrict to a family set:

```bash
./run-full-matrix.sh --tests mp,sb,iriw --stress params-smoke.txt
```

### Tune With MALLOC

```bash
./tune.sh all-tests.txt --mem-backend MALLOC
```

Without recompiling:

```bash
./tune.sh all-tests.txt --no-compile --mem-backend MALLOC
```

### MALLOC Notes

- `run-full-matrix.sh` already defaults to `MALLOC`, so `--mem-backend MALLOC` is optional there.
- Compiled MALLOC binaries end with `-malloc-runner`.
- The Makefile does not have a dedicated `tune-malloc` target, so use `make tune MEM_BACKEND=MALLOC`, `make tune-no-compile MEM_BACKEND=MALLOC`, or call `tune.sh` directly.

### 1. Compile The Whole Suite

```bash
make compile-only
```

Common variants:

```bash
make -j8 compile-only
make compile-only MEM_BACKEND=MANAGED
make compile-only TESTS=all-tests.txt HET_DEBUG=1
```

This produces binaries in `target/` with names like:

```text
target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-hostalloc-runner
```

Binary naming format:

```text
target/<test>-<tb>-<het>-<scope>-<fence_scope>-<variant>-<backend>-runner
```

The backend suffix is one of `hostalloc`, `managed`, or `malloc`.

### 2. Run A Single Compiled Litmus Binary Directly

Every compiled runner accepts the same flags:

| Flag | Meaning |
| --- | --- |
| `-s <file>` | Required stress-parameter file |
| `-t <file>` | Required test-parameter file from `params/` |
| `-x` | Print per-iteration weak-result details |

Example using the `mp` test:

```bash
./target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-hostalloc-runner \
  -s params-smoke.txt \
  -t params/2-loc.txt
```

Example with extra result printing:

```bash
./target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-hostalloc-runner \
  -s params-smoke.txt \
  -t params/2-loc.txt \
  -x
```

The runner prints a summary including:

- `Time taken`
- `Weak behavior rate`
- `Total behaviors`
- `Number of weak behaviors`

### 3. Sweep Many Configurations

Quick compile-and-run smoke sweep:

```bash
./run-full-matrix.sh --mode compile-and-run --stress params-smoke.txt -j 4
```

Compile only:

```bash
./run-full-matrix.sh --mode compile-only -j 8
```

Run only from an existing `target/`:

```bash
./run-full-matrix.sh --mode run-only --resume full-matrix-results/manual-rerun
```

Restrict the sweep to one test:

```bash
./run-full-matrix.sh --filter-test mp --stress params-smoke.txt
```

Restrict the sweep to several families:

```bash
./run-full-matrix.sh --mode compile-only --tests sb lb read store
```

### 4. Run The Infinite Tuning Loop

Compile then tune:

```bash
make tune
```

Tune without recompiling:

```bash
make tune-no-compile
```

Direct script form:

```bash
./tune.sh all-tests.txt --mem-backend HOSTALLOC
./tune.sh all-tests.txt --no-compile --mem-backend MANAGED
```

Stop the tuning loop with `Ctrl+C`.

## Tuning List Format

`all-tests.txt` is a list of tuning files, one per line, for example:

```text
tuning-files/mp.txt
tuning-files/wrc.txt
tuning-files/iriw.txt
```

Each tuning file defines the compile matrix for one test. For example, `tuning-files/mp.txt`:

```text
mp 2-loc.txt
TB_0_1 TB_01
HET_C0_G1 HET_C1_G0
SCOPE_SYSTEM SCOPE_DEVICE
ACQ_REL RELAXED
```

Line meanings:

| Line | Meaning |
| --- | --- |
| 1 | Test name and parameter file from `params/` |
| 2 | Thread-block topology macros |
| 3 | CPU/GPU split macros |
| 4 | scope macros |
| 5 | Non-fence variant macros |
| 6 | Optional fence-scope macros |
| 7 | Optional fence variant macros |

## Common Workflows

### Quick Sanity Check

```bash
make -j4 compile-only TESTS=all-tests.txt
./run-full-matrix.sh --mode run-only --stress params-smoke.txt --filter-test mp
```

### Compile And Run One Binary Manually

```bash
make compile-only TESTS=all-tests.txt MEM_BACKEND=HOSTALLOC
./target/mp-TB_0_1-HET_C0_G1-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-hostalloc-runner -s params-smoke.txt -t params/2-loc.txt
```

### Analyze Tuning Results

```bash
python3 analyze.py
python3 analyze.py results --csv results-summary.csv
python3 analyze.py --log tune.log
```

## Notes

- `HOSTALLOC` is the Makefile default memory backend.
- `MALLOC` is the `run-full-matrix.sh` default memory backend.
- `make help`, `./run-full-matrix.sh --help`, and `python3 analyze.py --help` reflect the current command-line surface and are the best source for future changes.
- The 2-thread catalog now includes `sb`, `lb`, `read`, and `store` in addition to `mp` and `2+2w`.
- For the full test catalog and supported split/topology combinations, see `docs/TEST-CATALOG.md`.
