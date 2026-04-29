#!/bin/bash

set -u

ROOT_DIR=$(dirname "$0")
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)

RESULT_DIR="$ROOT_DIR/full-matrix-results/ismm-87"
LOG_DIR="$RESULT_DIR/logs"
TARGET_DIR="$ROOT_DIR/target"
RESULT_CSV="$RESULT_DIR/results.csv"
RUNNER_SRC="$ROOT_DIR/ismm_runner.cu"
STRESS_FILE=${STRESS_FILE:-$ROOT_DIR/params-ismm.txt}
TEST_FILE=${TEST_FILE:-$ROOT_DIR/params/2-loc.txt}
NVCC=${NVCC:-/usr/local/cuda-12.4/bin/nvcc}
ARCH=${ARCH:-sm_90}
MEM_BACKEND=${MEM_BACKEND:-MALLOC}
HET_DEBUG=${HET_DEBUG:-0}
HOST_ARCH_FLAGS=${HOST_ARCH_FLAGS:--march=armv8.3-a+rcpc}

declare -a EXPERIMENTS=(
  "sb-arm-baseline"
  "sb-ptx-baseline"
  "sb-arm-rel-acq-rcsc"
  "sb-arm-rel-acq-rcpc"
  "sb-ptx-rel-acq-system"
  "sb-ptx-rel-acq-device"
  "sb-mixed-arm-ptx-t0-arm"
  "sb-mixed-arm-ptx-t0-ptx"
  "iriw-arm-baseline"
  "iriw-ptx-baseline"
  "iriw-arm-rel-acq-rcsc"
  "iriw-arm-rel-acq-rcpc"
  "iriw-ptx-rel-acq"
  "iriw-mixed-2rel-ptx-2acq-arm"
  "iriw-mixed-2acq-ptx-2rel-arm"
)

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
    echo "Invalid MEM_BACKEND: $MEM_BACKEND" >&2
    exit 1
    ;;
esac

RUNNER_BIN="$TARGET_DIR/ismm-runner-$MEM_SHORT"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$TARGET_DIR"

if [ ! -f "$RESULT_CSV" ]; then
  printf 'run_id,cycle,experiment,expected,time_seconds,weak_rate,total_behaviors,weak_behaviors,status,log_file\n' > "$RESULT_CSV"
fi

compile_runner() {
  local debug_flag=()
  if [ "$HET_DEBUG" = "1" ]; then
    debug_flag=(-DHET_DEBUG=1)
  fi

  "$NVCC" -D"$MEM_DEF" "${debug_flag[@]}" -I"$ROOT_DIR" -rdc=true -arch "$ARCH" \
    -Xcompiler "$HOST_ARCH_FLAGS" "$RUNNER_SRC" -o "$RUNNER_BIN" -diag-suppress 177
}

if [ ! -x "$RUNNER_BIN" ] || [ "$RUNNER_SRC" -nt "$RUNNER_BIN" ] || [ "$STRESS_FILE" -nt "$RUNNER_BIN" ]; then
  compile_runner
fi

completed_runs=$(( $(wc -l < "$RESULT_CSV") - 1 ))
if [ "$completed_runs" -lt 0 ]; then
  completed_runs=0
fi

count=${#EXPERIMENTS[@]}

while true; do
  start_index=$((completed_runs % count))
  offset=0
  while [ "$offset" -lt "$count" ]; do
    index=$(((start_index + offset) % count))
    experiment=${EXPERIMENTS[$index]}
    run_id=$((completed_runs + 1))
    cycle=$((completed_runs / count))
    log_file="$LOG_DIR/$(printf '%06d' "$run_id")-$experiment.log"

    output=$(
      "$RUNNER_BIN" -e "$experiment" -s "$STRESS_FILE" -t "$TEST_FILE" 2>&1 | tee "$log_file"
    )
    status=$?

    time_seconds=$(printf '%s\n' "$output" | sed -n 's/^Time taken: //p' | sed 's/ seconds$//' | tail -n 1)
    weak_rate=$(printf '%s\n' "$output" | sed -n 's/^Weak behavior rate: //p' | sed 's/ per second$//' | tail -n 1)
    total_behaviors=$(printf '%s\n' "$output" | sed -n 's/^Total behaviors: //p' | tail -n 1)
    weak_behaviors=$(printf '%s\n' "$output" | sed -n 's/^Number of weak behaviors: //p' | tail -n 1)
    expected=$(printf '%s\n' "$output" | sed -n 's/^Expectation: //p' | tail -n 1)

    if [ "$status" -eq 0 ]; then
      row_status="ok"
    else
      row_status="run_failed"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$run_id" "$cycle" "$experiment" "$expected" "$time_seconds" "$weak_rate" \
      "$total_behaviors" "$weak_behaviors" "$row_status" "$log_file" >> "$RESULT_CSV"

    completed_runs=$((completed_runs + 1))
    if [ "$status" -ne 0 ]; then
      echo "Experiment failed: $experiment" >&2
      exit "$status"
    fi

    offset=$((offset + 1))
  done
done
