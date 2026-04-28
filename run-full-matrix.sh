#!/bin/bash

set -u

usage() {
  cat <<'EOF'
Usage: ./run-full-matrix.sh [options]

Options:
  --mode <compile-and-run|compile-only|run-only>   Default: compile-and-run
  --tests <family[,family...]|family family...>    Default: all
  --mem-backend <HOSTALLOC|MANAGED|MALLOC>         Default: MALLOC
  --nvcc <path>                                    Default: /usr/local/cuda-12.4/bin/nvcc
  --arch <arch>                                    Default: sm_90
  --stress <file>                                  Default: params-smoke.txt
  -j, --jobs <count>                               Parallel compile workers, default: 1
  --out-dir <dir>                                  Default: full-matrix-results/<timestamp>
  --resume <dir>                                   Resume an existing output directory
  --target-dir <dir>                               Default: target
  --tests-file <file>                              Tuning-list file, default: all-tests.txt
  --filter-test <name>                             Only include one test name
  --help                                           Show this message

Notes:
  - `-j/--jobs` applies to the compile stage only.
  - Run execution remains serial so output logs stay ordered.

Examples:
  ./run-full-matrix.sh --mode compile-only -j 8
  ./run-full-matrix.sh --tests mp,iriw,sb --stress params-smoke.txt
  ./run-full-matrix.sh --mode compile-and-run --stress params-smoke.txt -j 4
  ./run-full-matrix.sh --mode run-only --resume full-matrix-results/manual-rerun
EOF
}

MODE="compile-and-run"
TESTS_FILE="all-tests.txt"
TEST_FAMILIES_RAW="all"
MEM_BACKEND="MALLOC"
NVCC="/usr/local/cuda-12.4/bin/nvcc"
ARCH="sm_90"
STRESS_FILE="params-smoke.txt"
TARGET_DIR="target"
FILTER_TEST=""
OUT_DIR=""
RESUME_DIR=""
JOBS=1
HET_DEBUG="${HET_DEBUG:-0}"
declare -a REQUESTED_TEST_FAMILIES=()
declare -A REQUESTED_TEST_FAMILY_SET=()
declare -A AVAILABLE_TEST_FAMILY_SET=()

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --tests)
      shift
      if [ $# -eq 0 ]; then
        echo "Missing value for --tests" >&2
        exit 1
      fi
      TEST_FAMILIES_RAW="$1"
      shift
      while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
        TEST_FAMILIES_RAW+=" $1"
        shift
      done
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
    --resume)
      RESUME_DIR="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --tests-file)
      TESTS_FILE="$2"
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

normalize_test_families() {
  local raw="$1"
  local item=""

  raw=${raw//,/ }
  for item in $raw; do
    item=$(printf '%s' "$item" | tr '[:upper:]' '[:lower:]' | xargs)
    [ -z "$item" ] && continue
    REQUESTED_TEST_FAMILIES+=("$item")
  done

  if [ ${#REQUESTED_TEST_FAMILIES[@]} -eq 0 ]; then
    REQUESTED_TEST_FAMILIES=("all")
  fi
}

load_available_test_families() {
  local tuning_file=""
  local lines=()
  local test_name=""

  while IFS= read -r tuning_file || [ -n "$tuning_file" ]; do
    tuning_file=$(printf '%s' "$tuning_file" | xargs)
    [ -z "$tuning_file" ] && continue
    [ ! -f "$tuning_file" ] && continue

    mapfile -t lines < "$tuning_file"
    read -r test_name _ <<< "${lines[0]}"
    [ -z "$test_name" ] && continue
    AVAILABLE_TEST_FAMILY_SET["$test_name"]=1
  done < "$TESTS_FILE"
}

should_include_test() {
  local test="$1"

  if [ ${#REQUESTED_TEST_FAMILY_SET[@]} -eq 0 ]; then
    return 0
  fi

  [ -n "${REQUESTED_TEST_FAMILY_SET[$test]+x}" ]
}

restore_csv_counts() {
  local csv_file="$1"
  if [ ! -f "$csv_file" ]; then
    echo 0
    return
  fi

  local lines
  lines=$(wc -l < "$csv_file")
  if [ "$lines" -le 1 ]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

restore_failure_counts() {
  local csv_file="$1"

  if [ ! -f "$csv_file" ]; then
    echo 0
    return
  fi

  awk -F, 'NR > 1 && $9 != "ok" { count++ } END { print count + 0 }' "$csv_file"
}

restore_manifest_count() {
  local manifest_file="$1"

  if [ ! -f "$manifest_file" ]; then
    echo 0
    return
  fi

  awk 'END { print NR + 0 }' "$manifest_file"
}

csv_has_index() {
  local csv_file="$1"
  local index="$2"

  grep -q "^${index}," "$csv_file"
}

append_csv_if_missing() {
  local csv_file="$1"
  local index="$2"
  local row="$3"

  if csv_has_index "$csv_file" "$index"; then
    return
  fi

  printf '%s\n' "$row" >> "$csv_file"
}

write_resume_metadata() {
  local metadata_file="$1"

  cat > "$metadata_file" <<EOF
mode=$MODE
tests_file=$TESTS_FILE
tests=$TEST_FAMILIES_RAW
filter_test=$FILTER_TEST
mem_backend=$MEM_BACKEND
target_dir=$TARGET_DIR
arch=$ARCH
stress_file=$STRESS_FILE
EOF
}

validate_resume_metadata() {
  local metadata_file="$1"
  local key=""
  local value=""

  [ ! -f "$metadata_file" ] && return

  while IFS='=' read -r key value || [ -n "$key" ]; do
    case "$key" in
      mode)
        [ "$MODE" = "$value" ] || { echo "Resume directory was created with --mode $value" >&2; exit 1; }
        ;;
      tests_file)
        [ "$TESTS_FILE" = "$value" ] || { echo "Resume directory was created with --tests-file $value" >&2; exit 1; }
        ;;
      tests)
        [ "$TEST_FAMILIES_RAW" = "$value" ] || { echo "Resume directory was created with --tests '$value'" >&2; exit 1; }
        ;;
      filter_test)
        [ "$FILTER_TEST" = "$value" ] || { echo "Resume directory was created with --filter-test '$value'" >&2; exit 1; }
        ;;
      mem_backend)
        [ "$MEM_BACKEND" = "$value" ] || { echo "Resume directory was created with --mem-backend $value" >&2; exit 1; }
        ;;
      target_dir)
        [ "$TARGET_DIR" = "$value" ] || { echo "Resume directory was created with --target-dir $value" >&2; exit 1; }
        ;;
      arch)
        [ "$ARCH" = "$value" ] || { echo "Resume directory was created with --arch $value" >&2; exit 1; }
        ;;
      stress_file)
        [ "$STRESS_FILE" = "$value" ] || { echo "Resume directory was created with --stress $value" >&2; exit 1; }
        ;;
    esac
  done < "$metadata_file"
}

normalize_test_families "$TEST_FAMILIES_RAW"

if [ -n "$RESUME_DIR" ]; then
  if [ -n "$OUT_DIR" ] && [ "$OUT_DIR" != "$RESUME_DIR" ]; then
    echo "--out-dir and --resume must point to the same directory when both are provided" >&2
    exit 1
  fi
  if [ ! -d "$RESUME_DIR" ]; then
    echo "Resume directory not found: $RESUME_DIR" >&2
    exit 1
  fi
  OUT_DIR="$RESUME_DIR"
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
RESUME_METADATA="$OUT_DIR/run-full-matrix.meta"

if [ -z "$RESUME_DIR" ]; then
  : > "$MANIFEST"
  : > "$COMPILE_LOG"
  : > "$RUN_LOG"
  printf 'index,test,tb,het,scope,fence_scope,variant,binary,status\n' > "$COMPILE_CSV"
  printf 'index,test,tb,het,scope,fence_scope,variant,binary,status,total_behaviors,weak_behaviors\n' > "$RUN_CSV"
else
  touch "$MANIFEST" "$COMPILE_LOG" "$RUN_LOG"
  if [ ! -f "$COMPILE_CSV" ]; then
    printf 'index,test,tb,het,scope,fence_scope,variant,binary,status\n' > "$COMPILE_CSV"
  fi
  if [ ! -f "$RUN_CSV" ]; then
    printf 'index,test,tb,het,scope,fence_scope,variant,binary,status,total_behaviors,weak_behaviors\n' > "$RUN_CSV"
  fi
fi

mkdir -p "$COMPILE_STATUS_DIR"

if [ -n "$RESUME_DIR" ]; then
  validate_resume_metadata "$RESUME_METADATA"
else
  write_resume_metadata "$RESUME_METADATA"
fi

compile_total=$(restore_csv_counts "$COMPILE_CSV")
compile_failures=$(restore_failure_counts "$COMPILE_CSV")
run_total=$(restore_csv_counts "$RUN_CSV")
run_failures=$(restore_failure_counts "$RUN_CSV")
manifest_total=0

load_available_test_families

for requested_test in "${REQUESTED_TEST_FAMILIES[@]}"; do
  if [ "$requested_test" = "all" ]; then
    REQUESTED_TEST_FAMILY_SET=()
    break
  fi
  if [ -z "${AVAILABLE_TEST_FAMILY_SET[$requested_test]+x}" ]; then
    echo "Unknown test family: $requested_test" >&2
    echo "Available test families: $(printf '%s\n' "${!AVAILABLE_TEST_FAMILY_SET[@]}" | sort | paste -sd ', ' -)" >&2
    exit 1
  fi
  REQUESTED_TEST_FAMILY_SET["$requested_test"]=1
done

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

    if ! should_include_test "$test"; then
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
  local status=""
  local had_csv_row=0
  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    if [ -f "$COMPILE_STATUS_DIR/$index.status" ]; then
      had_csv_row=0
      if csv_has_index "$COMPILE_CSV" "$index"; then
        had_csv_row=1
      fi
      status=$(cat "$COMPILE_STATUS_DIR/$index.status")
      append_csv_if_missing "$COMPILE_CSV" "$index" \
        "$index,$test,$tb,$het,$scope,$fence_scope,$variant,$binary,$status"
      if [ "$status" != "ok" ] && [ "$had_csv_row" -eq 0 ]; then
        compile_failures=$((compile_failures + 1))
      fi
      continue
    fi

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
  local status=""
  local had_csv_row=0
  local -a queued_indexes=()
  mkdir -p "$COMPILE_JOB_DIR"

  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    if [ -f "$COMPILE_STATUS_DIR/$index.status" ]; then
      had_csv_row=0
      if csv_has_index "$COMPILE_CSV" "$index"; then
        had_csv_row=1
      fi
      status=$(cat "$COMPILE_STATUS_DIR/$index.status")
      append_csv_if_missing "$COMPILE_CSV" "$index" \
        "$index,$test,$tb,$het,$scope,$fence_scope,$variant,$binary,$status"
      if [ "$status" != "ok" ] && [ "$had_csv_row" -eq 0 ]; then
        compile_failures=$((compile_failures + 1))
      fi
      continue
    fi

    compile_total=$((compile_total + 1))
    queued=$((queued + 1))
    queued_indexes+=("$index")
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

  local index=""
  for index in "${queued_indexes[@]}"; do
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
  local status=""
  while IFS=$'\t' read -r index test params tb het scope fence_scope variant binary; do
    if csv_has_index "$RUN_CSV" "$index"; then
      continue
    fi

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

if [ -n "$RESUME_DIR" ] && [ -s "$MANIFEST" ]; then
  manifest_total=$(restore_manifest_count "$MANIFEST")
else
  build_manifest
fi

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
