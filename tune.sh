#!/bin/bash

# Heterogeneous CPU-GPU litmus test tuning script
# Adapted from cuda-litmus tune.sh for het framework
#
# Usage:
#   ./tune.sh <tuning-list-file> [--no-compile] [--mem-backend <HOSTALLOC|MANAGED|MALLOC>]
#
# The tuning list file contains paths to individual tuning files, one per line.
# Each tuning file has the format:
#   Line 1: test_name param_file
#   Line 2: TB configs (space-separated)
#   Line 3: HET split macros (space-separated)
#   Line 4: SCOPE configs (space-separated)
#   Line 5: Variants / non-fence (space-separated)
#   Line 6 (optional): FENCE_SCOPE options
#   Line 7 (optional): Fence variants

PARAM_FILE="params.txt"
RESULT_DIR="results"
SHADER_DIR="kernels"
PARAMS_DIR="params"
TARGET_DIR="target"
NVCC="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
ARCH="${ARCH:-sm_90}"
HET_DEBUG="${HET_DEBUG:-0}"

# Default memory backend
MEM_BACKEND="MEM_HOSTALLOC"
COMPILE=true

# =============================================================================
# Argument parsing
# =============================================================================

if [ $# -lt 1 ]; then
  echo "Usage: $0 <tuning-list-file> [--no-compile] [--mem-backend <HOSTALLOC|MANAGED|MALLOC>]"
  exit 1
fi

tuning_file=$1
shift

while [ $# -gt 0 ]; do
  case "$1" in
    --no-compile)
      COMPILE=false
      shift
      ;;
    --mem-backend)
      shift
      case "$1" in
        HOSTALLOC) MEM_BACKEND="MEM_HOSTALLOC" ;;
        MANAGED)   MEM_BACKEND="MEM_MANAGED" ;;
        MALLOC)    MEM_BACKEND="MEM_MALLOC" ;;
        *)
          echo "Unknown memory backend: $1 (expected HOSTALLOC, MANAGED, or MALLOC)"
          exit 1
          ;;
      esac
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Short name for memory backend (used in binary and result directory names)
case "$MEM_BACKEND" in
  MEM_HOSTALLOC) MEM_SHORT="hostalloc" ;;
  MEM_MANAGED)   MEM_SHORT="managed" ;;
  MEM_MALLOC)    MEM_SHORT="malloc" ;;
esac

case "$HET_DEBUG" in
  0|1) ;;
  *)
    echo "Invalid HET_DEBUG value: $HET_DEBUG (expected 0 or 1)"
    exit 1
    ;;
esac

# =============================================================================
# Utility functions
# =============================================================================

function make_even() {
  if (( $1 % 2 == 0 )); then
    echo "$1"
  else
    echo "$(($1 + 1))"
  fi
}

function random_between() {
  local min=$1
  local max=$2
  local range=$((max - min + 1))
  local random=$((RANDOM % range + min))
  echo "$random"
}

# =============================================================================
# Random configuration generator (extended for het)
# =============================================================================

function random_config() {
  local workgroupLimiter=$1
  local workgroupSizeLimiter=$2

  echo "testIterations=1000" > $PARAM_FILE
  local testingWorkgroups=$(random_between 4 $workgroupLimiter)
  echo "testingWorkgroups=$testingWorkgroups" >> $PARAM_FILE
  local maxWorkgroups=$(random_between $testingWorkgroups $workgroupLimiter)
  echo "maxWorkgroups=$maxWorkgroups" >> $PARAM_FILE
  # ensures total threads is divisible by 2
  local workgroupSize=$(make_even $(random_between 1 $workgroupSizeLimiter))
  echo "workgroupSize=$workgroupSize" >> $PARAM_FILE
  echo "shufflePct=$(random_between 0 100)" >> $PARAM_FILE
  echo "barrierPct=$(random_between 0 100)" >> $PARAM_FILE
  local stressLineSize=$(echo "$(random_between 2 10)^2" | bc)
  echo "stressLineSize=$stressLineSize" >> $PARAM_FILE
  local stressTargetLines=$(random_between 1 16)
  echo "stressTargetLines=$stressTargetLines" >> $PARAM_FILE
  echo "scratchMemorySize=$((32 * $stressLineSize * $stressTargetLines))" >> $PARAM_FILE
  echo "memStride=$(random_between 1 7)" >> $PARAM_FILE
  echo "memStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "memStressIterations=$(random_between 0 1024)" >> $PARAM_FILE
  echo "memStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "preStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "preStressIterations=$(random_between 0 128)" >> $PARAM_FILE
  echo "preStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "stressAssignmentStrategy=$(random_between 0 1)" >> $PARAM_FILE
  echo "permuteThread=419" >> $PARAM_FILE

  # Het-specific params
  echo "cpuStressThreads=$(random_between 0 16)" >> $PARAM_FILE
  echo "cpuPreStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "cpuPreStressIterations=$(random_between 0 128)" >> $PARAM_FILE
  echo "cpuPreStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "barrierSpinLimit=$(random_between 1024 8192)" >> $PARAM_FILE
}

# =============================================================================
# Run a single test binary with current params
# =============================================================================

function run_test() {
  local test=$1
  local tb=$2
  local het=$3
  local scope=$4
  local fence_scope=$5
  local variant=$6
  local params=$7

  local binary_name="$test-$tb-$het-$scope-$fence_scope-$variant-$MEM_SHORT-runner"
  local result_key="$test-$tb-$het-$scope-$fence_scope-$variant-$MEM_SHORT"

  if [ ! -f "$TARGET_DIR/$binary_name" ]; then
    echo "  SKIP $result_key (binary not found)"
    return
  fi

  res=$(./$TARGET_DIR/$binary_name -s $PARAM_FILE -t $PARAMS_DIR/$params 2>&1)
  local exit_code=$?

  if [ $exit_code -ne 0 ]; then
    echo "  FAIL $result_key (exit code $exit_code)"
    return
  fi

  local weak_behaviors=$(printf '%s\n' "$res" | sed -n 's/^Number of weak behaviors: //p' | tail -n 1)
  local total_behaviors=$(printf '%s\n' "$res" | sed -n 's/^Total behaviors: //p' | tail -n 1)
  local weak_rate=$(printf '%s\n' "$res" | sed -n 's/^Weak behavior rate: //p' | sed 's/ per second$//' | tail -n 1)

  if [ -z "$weak_behaviors" ] || [ -z "$total_behaviors" ] || [ -z "$weak_rate" ]; then
    echo "  FAIL $result_key (could not parse runner summary)"
    echo "  Runner output follows:"
    printf '%s\n' "$res"
    return 1
  fi

  echo "  $result_key  weak: $weak_behaviors, total: $total_behaviors, rate: $weak_rate/s"

  if (( $(echo "$weak_rate > 0" | bc -l) )); then
    local test_result_dir="$RESULT_DIR/$result_key"
    if [ ! -d "$test_result_dir" ]; then
      mkdir -p "$test_result_dir"
      cp $PARAM_FILE "$test_result_dir"
      echo "$weak_rate" > "$test_result_dir/rate"
      echo "$weak_behaviors" > "$test_result_dir/weak"
      echo "$total_behaviors" > "$test_result_dir/total"
    else
      local max_rate=$(cat "$test_result_dir/rate")
      if (( $(echo "$weak_rate > $max_rate" | bc -l) )); then
        cp $PARAM_FILE "$test_result_dir"
        echo "$weak_rate" > "$test_result_dir/rate"
        echo "$weak_behaviors" > "$test_result_dir/weak"
        echo "$total_behaviors" > "$test_result_dir/total"
      fi
    fi
  fi
}

# =============================================================================
# Create output directories
# =============================================================================

if [ ! -d "$RESULT_DIR" ]; then
  mkdir -p $RESULT_DIR
fi

if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p $TARGET_DIR
fi

# =============================================================================
# Read the list of tuning files
# =============================================================================

readarray test_files < $tuning_file

# =============================================================================
# Compilation phase
# =============================================================================

if "$COMPILE"; then
  echo "============================================"
  echo "Compilation phase (memory backend: $MEM_BACKEND)"
  echo "============================================"

  for test_file in "${test_files[@]}"; do
    # Trim whitespace/newline
    test_file=$(echo "$test_file" | xargs)
    [ -z "$test_file" ] && continue
    [ ! -f "$test_file" ] && { echo "WARNING: tuning file '$test_file' not found, skipping"; continue; }

    read -a test_info <<< "$(sed -n '1p' "$test_file")"
    read -a threadblocks <<< "$(sed -n '2p' "$test_file")"
    read -a het_splits <<< "$(sed -n '3p' "$test_file")"
    read -a scopes <<< "$(sed -n '4p' "$test_file")"
    read -a variants <<< "$(sed -n '5p' "$test_file")"

    local_lines=$(wc -l < "$test_file")
    has_fences=false
    if [[ $local_lines -ge 7 ]]; then
      has_fences=true
      read -a fence_scopes <<< "$(sed -n '6p' "$test_file")"
      read -a fence_variants <<< "$(sed -n '7p' "$test_file")"
    fi

    test="${test_info[0]}"

    # Compile non-fence variant combinations
    for tb in "${threadblocks[@]}"; do
      for het in "${het_splits[@]}"; do
        for scope in "${scopes[@]}"; do
          for variant in "${variants[@]}"; do
            binary_name="$test-$tb-$het-$scope-NO_FENCE-$variant-$MEM_SHORT-runner"
            echo "Compiling $binary_name"
            debug_def=()
            if [ "$HET_DEBUG" = "1" ]; then
              debug_def=(-DHET_DEBUG=1)
            fi
            "$NVCC" -D$tb -D$het -D$scope -D$variant -D$MEM_BACKEND "${debug_def[@]}" \
                 -I. -rdc=true -arch $ARCH \
                 runner.cu "kernels/$test.cu" \
                 -o "$TARGET_DIR/$binary_name" 2>&1
            if [ $? -ne 0 ]; then
              echo "  COMPILATION FAILED: $binary_name"
            fi
          done

          if "$has_fences"; then
            for f_scope in "${fence_scopes[@]}"; do
              for f_variant in "${fence_variants[@]}"; do
                binary_name="$test-$tb-$het-$scope-$f_scope-$f_variant-$MEM_SHORT-runner"
                echo "Compiling $binary_name"
                debug_def=()
                if [ "$HET_DEBUG" = "1" ]; then
                  debug_def=(-DHET_DEBUG=1)
                fi
                "$NVCC" -D$tb -D$het -D$scope -D$f_scope -D$f_variant -D$MEM_BACKEND "${debug_def[@]}" \
                     -I. -rdc=true -arch $ARCH \
                     runner.cu "kernels/$test.cu" \
                     -o "$TARGET_DIR/$binary_name" 2>&1
                if [ $? -ne 0 ]; then
                  echo "  COMPILATION FAILED: $binary_name"
                fi
              done
            done
          fi
        done
      done
    done
  done

  echo ""
  echo "Compilation complete."
  echo ""
fi

# =============================================================================
# Tuning loop (infinite — Ctrl+C to stop)
# =============================================================================

iter=0

while true; do
  echo "============================================"
  echo "Iteration: $iter"
  echo "============================================"
  random_config 1024 256

  for test_file in "${test_files[@]}"; do
    # Trim whitespace/newline
    test_file=$(echo "$test_file" | xargs)
    [ -z "$test_file" ] && continue
    [ ! -f "$test_file" ] && continue

    read -a test_info <<< "$(sed -n '1p' "$test_file")"
    read -a threadblocks <<< "$(sed -n '2p' "$test_file")"
    read -a het_splits <<< "$(sed -n '3p' "$test_file")"
    read -a scopes <<< "$(sed -n '4p' "$test_file")"
    read -a variants <<< "$(sed -n '5p' "$test_file")"

    local_lines=$(wc -l < "$test_file")
    has_fences=false
    if [[ $local_lines -ge 7 ]]; then
      has_fences=true
      read -a fence_scopes <<< "$(sed -n '6p' "$test_file")"
      read -a fence_variants <<< "$(sed -n '7p' "$test_file")"
    fi

    test="${test_info[0]}"
    params="${test_info[1]}"

    echo "--- $test ---"

    for tb in "${threadblocks[@]}"; do
      for het in "${het_splits[@]}"; do
        for scope in "${scopes[@]}"; do
          for variant in "${variants[@]}"; do
            run_test "$test" "$tb" "$het" "$scope" "NO_FENCE" "$variant" "$params"
          done

          if "$has_fences"; then
            for f_scope in "${fence_scopes[@]}"; do
              for f_variant in "${fence_variants[@]}"; do
                run_test "$test" "$tb" "$het" "$scope" "$f_scope" "$f_variant" "$params"
              done
            done
          fi
        done
      done
    done
  done

  iter=$((iter + 1))
done
