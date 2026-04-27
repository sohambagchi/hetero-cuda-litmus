#!/bin/bash

set -u

usage() {
  cat <<'EOF'
Usage: ./run-full-matrix.sh [options]

Options:
  --mode <compile-and-run|compile-only|run-only>   Default: compile-and-run
  --tests <file>                                   Default: all-tests.txt
  --mem-backend <HOSTALLOC|MANAGED|MALLOC>         Default: MALLOC
  --nvcc <path>                                    Default: /usr/local/cuda-12.4/bin/nvcc
  --arch <arch>                                    Default: sm_90
  --stress <file>                                  Default: params-smoke.txt
  -j, --jobs <count>                               Parallel compile workers, default: 1
  --out-dir <dir>                                  Default: full-matrix-results/<timestamp>
  --target-dir <dir>                               Default: target
  --filter-test <name>                             Only include one test name
  --help                                           Show this message

Notes:
  - `-j/--jobs` applies to the compile stage only.
  - Run execution remains serial so output logs stay ordered.

Examples:
  ./run-full-matrix.sh --mode compile-only -j 8
  ./run-full-matrix.sh --mode compile-and-run --stress params-smoke.txt -j 4
  ./run-full-matrix.sh --mode run-only --out-dir full-matrix-results/manual-rerun
EOF
}

MODE="compile-and-run"
TESTS_FILE="all-tests.txt"
MEM_BACKEND="MALLOC"
NVCC="/usr/local/cuda-12.4/bin/nvcc"
ARCH="sm_90"
STRESS_FILE="params-smoke.txt"
TARGET_DIR="target"
FILTER_TEST=""
OUT_DIR=""
JOBS=1
HET_DEBUG="${HET_DEBUG:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --tests)
      TESTS_FILE="$2"
      shift 2
      ;;
    --mem-backend)
      MEM_BACKEND="$2"
      shift 2
      ;;
    --nvcc)
      NVCC="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --stress)
      STRESS_FILE="$2"
      shift 2
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --filter-test)
      FILTER_TEST="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$MODE" in
  compile-and-run|compile-only|run-only)
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    exit 1
    ;;
esac

case "$MEM_BACKEND" in
  HOSTALLOC)
    MEM_DEF="MEM_HOSTALLOC"
    MEM_SHORT="hostalloc"
    ;;
  MANAGED)
    MEM_DEF="MEM_MANAGED"
    MEM_SHORT="managed"
    ;;
  MALLOC)
    MEM_DEF="MEM_MALLOC"
    MEM_SHORT="malloc"
    ;;
  *)
    echo "Invalid memory backend: $MEM_BACKEND" >&2
    exit 1
    ;;
esac

case "$JOBS" in
  ''|*[!0-9]*)
    echo "Invalid job count: $JOBS" >&2
    exit 1
    ;;
esac

case "$HET_DEBUG" in
  0|1)
    ;;
  *)
    echo "Invalid HET_DEBUG value: $HET_DEBUG (expected 0 or 1)" >&2
    exit 1
    ;;
esac

if [ "$JOBS" -lt 1 ]; then
  echo "Job count must be at least 1" >&2
  exit 1
fi

if [ ! -f "$TESTS_FILE" ]; then
  echo "Tests file not found: $TESTS_FILE" >&2
  exit 1
fi

if [ ! -f "$STRESS_FILE" ]; then
  echo "Stress file not found: $STRESS_FILE" >&2
  exit 1
fi

if [ -z "$OUT_DIR" ]; then
  OUT_DIR="full-matrix-results/$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$OUT_DIR" "$TARGET_DIR"

MANIFEST="$OUT_DIR/matrix.tsv"
COMPILE_LOG="$OUT_DIR/compile.log"
RUN_LOG="$OUT_DIR/run.log"
COMPILE_CSV="$OUT_DIR/compile-summary.csv"
RUN_CSV="$OUT_DIR/run-summary.csv"
COMPILE_STATUS_DIR="$OUT_DIR/compile-status"
COMPILE_JOB_DIR="$OUT_DIR/compile-jobs"

: > "$MANIFEST"
: > "$COMPILE_LOG"
: > "$RUN_LOG"
printf 'index,test,tb,het,scope,fence_scope,variant,binary,status\n' > "$COMPILE_CSV"
printf 'index,test,tb,het,scope,fence_scope,variant,binary,status,total_behaviors,weak_behaviors\n' > "$RUN_CSV"

mkdir -p "$COMPILE_STATUS_DIR"

compile_total=0
compile_failures=0
run_total=0
run_failures=0
manifest_total=0

run_compile_command() {
  local test="$1"
  local tb="$2"
  local het="$3"
  local scope="$4"
  local fence_scope="$5"
  local variant="$6"
  local binary="$7"

  local -a cmd=("$NVCC" "-D${tb}" "-D${het}" "-D${scope}")
  if [ "$fence_scope" != "NO_FENCE" ]; then
    cmd+=("-D${fence_scope}")
  fi
  cmd+=("-D${variant}" "-D${MEM_DEF}")
  if [ "$HET_DEBUG" = "1" ]; then
    cmd+=("-DHET_DEBUG=1")
  fi
  cmd+=(-I. -rdc=true -arch "$ARCH" runner.cu "kernels/${test}.cu" -o "$binary" -diag-suppress 177)

  "${cmd[@]}"
}

append_manifest() {
  local index="$1"
  local test="$2"
  local params="$3"
  local tb="$4"
  local het="$5"
  local scope="$6"
  local fence_scope="$7"
  local variant="$8"
  local binary="$9"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$index" "$test" "$params" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" >> "$MANIFEST"
}

build_manifest() {
  local index=0

  while IFS= read -r tuning_file || [ -n "$tuning_file" ]; do
    tuning_file=$(printf '%s' "$tuning_file" | xargs)
    [ -z "$tuning_file" ] && continue
    [ ! -f "$tuning_file" ] && { echo "Skipping missing tuning file: $tuning_file" | tee -a "$COMPILE_LOG" "$RUN_LOG"; continue; }

    mapfile -t lines < "$tuning_file"
    read -r test params <<< "${lines[0]}"

    if [ -n "$FILTER_TEST" ] && [ "$test" != "$FILTER_TEST" ]; then
      continue
    fi

    read -ra tbs <<< "${lines[1]}"
    read -ra hets <<< "${lines[2]}"
    read -ra scopes <<< "${lines[3]}"
    read -ra variants <<< "${lines[4]}"

    has_fences=false
    if [ ${#lines[@]} -ge 7 ]; then
      has_fences=true
      read -ra fence_scopes <<< "${lines[5]}"
      read -ra fence_variants <<< "${lines[6]}"
    fi

    for tb in "${tbs[@]}"; do
      for het in "${hets[@]}"; do
        for scope in "${scopes[@]}"; do
          for variant in "${variants[@]}"; do
            index=$((index + 1))
            append_manifest "$index" "$test" "$params" "$tb" "$het" "$scope" "NO_FENCE" "$variant" \
              "$TARGET_DIR/${test}-${tb}-${het}-${scope}-NO_FENCE-${variant}-${MEM_SHORT}-runner"
          done

          if $has_fences; then
            for fence_scope in "${fence_scopes[@]}"; do
              for variant in "${fence_variants[@]}"; do
                index=$((index + 1))
                append_manifest "$index" "$test" "$params" "$tb" "$het" "$scope" "$fence_scope" "$variant" \
                  "$TARGET_DIR/${test}-${tb}-${het}-${scope}-${fence_scope}-${variant}-${MEM_SHORT}-runner"
              done
            done
          fi
        done
      done
    done
  done < "$TESTS_FILE"

  manifest_total="$index"
}

compile_manifest_serial() {
  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    compile_total=$((compile_total + 1))
    printf '[compile %d/%d] %s\n' "$compile_total" "$manifest_total" "$binary" | tee -a "$COMPILE_LOG"

    if run_compile_command "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" >> "$COMPILE_LOG" 2>&1; then
      status="ok"
    else
      status="compile_failed"
      compile_failures=$((compile_failures + 1))
    fi

    printf '%s\n' "$status" > "$COMPILE_STATUS_DIR/$index.status"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$index" "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" "$status" >> "$COMPILE_CSV"
  done < "$MANIFEST"
}

compile_manifest_parallel() {
  local queued=0
  mkdir -p "$COMPILE_JOB_DIR"

  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    compile_total=$((compile_total + 1))
    queued=$((queued + 1))
    printf '[compile %d/%d] %s\n' "$compile_total" "$manifest_total" "$binary" | tee -a "$COMPILE_LOG"

    while [ "$(jobs -pr | wc -l)" -ge "$JOBS" ]; do
      wait -n
    done

    (
      if run_compile_command "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" > "$COMPILE_JOB_DIR/$index.log" 2>&1; then
        status="ok"
      else
        status="compile_failed"
      fi

      printf '%s\n' "$status" > "$COMPILE_STATUS_DIR/$index.status"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$index" "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" "$status" > "$COMPILE_JOB_DIR/$index.csv"
    ) &
  done < "$MANIFEST"

  wait

  local index=1
  while [ "$index" -le "$queued" ]; do
    if [ -f "$COMPILE_JOB_DIR/$index.log" ]; then
      cat "$COMPILE_JOB_DIR/$index.log" >> "$COMPILE_LOG"
    fi

    if [ -f "$COMPILE_STATUS_DIR/$index.status" ]; then
      status=$(cat "$COMPILE_STATUS_DIR/$index.status")
      if [ "$status" != "ok" ]; then
        compile_failures=$((compile_failures + 1))
      fi
    fi

    if [ -f "$COMPILE_JOB_DIR/$index.csv" ]; then
      cat "$COMPILE_JOB_DIR/$index.csv" >> "$COMPILE_CSV"
    fi

    index=$((index + 1))
  done
}

run_combo() {
  local index="$1"
  local test="$2"
  local params="$3"
  local tb="$4"
  local het="$5"
  local scope="$6"
  local fence_scope="$7"
  local variant="$8"
  local binary="$9"

  run_total=$((run_total + 1))
  printf '[run %d/%d] %s\n' "$run_total" "$manifest_total" "$binary" | tee -a "$RUN_LOG"

  if [ ! -x "$binary" ]; then
    run_failures=$((run_failures + 1))
    printf '%s,%s,%s,%s,%s,%s,%s,%s,missing,,\n' \
      "$index" "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" >> "$RUN_CSV"
    return 1
  fi

  local output
  if ! output=$("$binary" -s "$STRESS_FILE" -t "params/${params}" 2>&1); then
    printf '%s\n' "$output" >> "$RUN_LOG"
    run_failures=$((run_failures + 1))
    printf '%s,%s,%s,%s,%s,%s,%s,%s,run_failed,,\n' \
      "$index" "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" >> "$RUN_CSV"
    return 1
  fi

  printf '%s\n' "$output" >> "$RUN_LOG"

  local total_behaviors=""
  local weak_behaviors=""
  total_behaviors=$(printf '%s\n' "$output" | sed -n 's/^Total behaviors: //p' | tail -n 1)
  weak_behaviors=$(printf '%s\n' "$output" | sed -n 's/^Number of weak behaviors: //p' | tail -n 1)

  printf '%s,%s,%s,%s,%s,%s,%s,%s,ok,%s,%s\n' \
    "$index" "$test" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary" "$total_behaviors" "$weak_behaviors" >> "$RUN_CSV"
  return 0
}

run_manifest() {
  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    if [ "$MODE" = "compile-and-run" ]; then
      if [ ! -f "$COMPILE_STATUS_DIR/$index.status" ]; then
        continue
      fi

      status=$(cat "$COMPILE_STATUS_DIR/$index.status")
      if [ "$status" != "ok" ]; then
        continue
      fi
    fi

    run_combo "$index" "$test" "$params" "$tb" "$het" "$scope" "$fence_scope" "$variant" "$binary"
  done < "$MANIFEST"
}

build_manifest

if [ "$manifest_total" -eq 0 ]; then
  echo "No matrix entries matched the requested filters." >&2
  exit 1
fi

if [ "$MODE" != "run-only" ]; then
  if [ "$JOBS" -le 1 ]; then
    compile_manifest_serial
  else
    compile_manifest_parallel
  fi
fi

if [ "$MODE" != "compile-only" ]; then
  run_manifest
fi

cat <<EOF
Full matrix run complete.
Mode: $MODE
Compile jobs: $JOBS
HET_DEBUG: $HET_DEBUG
Output directory: $OUT_DIR
Matrix entries: $manifest_total
Compile attempts: $compile_total
Compile failures: $compile_failures
Run attempts: $run_total
Run failures: $run_failures
Compile summary: $COMPILE_CSV
Run summary: $RUN_CSV
EOF
