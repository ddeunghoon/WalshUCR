#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/wh_md_walsh_degree1_sweep.py"

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required but was not found on PATH." >&2
  exit 1
fi

N_SYS="${N_SYS:-3}"
M_LIST="${M_LIST:-5 6 7 8 9 10 11 12}"
INSTANCE_IDS="${INSTANCE_IDS:-0 1 2 3 4 5 6 7 8 9}"
NUM_RESTARTS="${NUM_RESTARTS:-50}"
SEED_START="${SEED_START:-0}"
STEPS="${STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
THRESHOLD="${THRESHOLD:-1e-6}"
TOL="${TOL:-5e-4}"
SU_DEPTH="${SU_DEPTH:-14}"
SCALE_INIT="${SCALE_INIT:-1.0}"
BIAS_SCALE_INIT="${BIAS_SCALE_INIT:-1.0}"
PROJECTION_STRATEGY="${PROJECTION_STRATEGY:-drop_extra}"
PLOT_DPI="${PLOT_DPI:-180}"
DISPLAY_FLOOR="${DISPLAY_FLOOR:-1e-7}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/results/wh_md_walsh_degree1_nsys3_scale1_drop_extra_restart_reuse_i10_r50}"
FINAL_ROOT="${FINAL_ROOT:-$RUN_ROOT/final}"
PARALLEL_PROGRESS="${PARALLEL_PROGRESS:-off}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

read -r -a M_LIST_ARR <<< "$M_LIST"
read -r -a INSTANCE_ID_ARR <<< "$INSTANCE_IDS"

if [ "${#M_LIST_ARR[@]}" -eq 0 ]; then
  echo "M_LIST must contain at least one M value." >&2
  exit 1
fi
if [ "${#INSTANCE_ID_ARR[@]}" -eq 0 ]; then
  echo "INSTANCE_IDS must contain at least one instance id." >&2
  exit 1
fi

if command -v nproc >/dev/null 2>&1; then
  CPU_COUNT="$(nproc)"
else
  CPU_COUNT=1
fi
DEFAULT_JOBS="$(( CPU_COUNT / 2 ))"
if [ "$DEFAULT_JOBS" -lt 1 ]; then
  DEFAULT_JOBS=1
fi
if [ "$DEFAULT_JOBS" -gt 8 ]; then
  DEFAULT_JOBS=8
fi
JOBS="${JOBS:-$DEFAULT_JOBS}"

SHARD_ROOT="$RUN_ROOT/shards"
LOG_ROOT="$RUN_ROOT/parallel_logs"
JOBLOG_PATH="$RUN_ROOT/parallel_joblog.tsv"
JOBLIST_PATH="$RUN_ROOT/joblist.tsv"

mkdir -p "$SHARD_ROOT" "$FINAL_ROOT" "$LOG_ROOT"
rm -f "$JOBLIST_PATH"
for M in "${M_LIST_ARR[@]}"; do
  for INSTANCE_ID in "${INSTANCE_ID_ARR[@]}"; do
    SHARD_DIR="$SHARD_ROOT/M${M}_instance$(printf '%02d' "$INSTANCE_ID")"
    printf "%s\t%s\t%s\n" "$M" "$INSTANCE_ID" "$SHARD_DIR" >> "$JOBLIST_PATH"
  done
done

PARALLEL_UI_ARGS=()
case "$PARALLEL_PROGRESS" in
  auto)
    if [ -t 2 ]; then
      PARALLEL_UI_ARGS+=(--bar)
    fi
    ;;
  bar)
    PARALLEL_UI_ARGS+=(--bar)
    ;;
  off)
    ;;
  *)
    echo "Unsupported PARALLEL_PROGRESS='$PARALLEL_PROGRESS'. Use auto, bar, or off." >&2
    exit 1
    ;;
esac

echo "run_root=$RUN_ROOT"
echo "final_root=$FINAL_ROOT"
echo "n_sys=$N_SYS"
echo "m_list=${M_LIST_ARR[*]}"
echo "instance_ids=${INSTANCE_ID_ARR[*]}"
echo "num_restarts=$NUM_RESTARTS"
echo "seed_start=$SEED_START"
echo "steps=$STEPS"
echo "eval_interval=$EVAL_INTERVAL"
echo "scale_init=$SCALE_INIT"
echo "bias_scale_init=$BIAS_SCALE_INIT"
echo "projection_strategy=$PROJECTION_STRATEGY"
echo "jobs=$JOBS"

parallel \
  --will-cite \
  --colsep '\t' \
  --jobs "$JOBS" \
  "${PARALLEL_UI_ARGS[@]}" \
  --joblog "$JOBLOG_PATH" \
  --results "$LOG_ROOT" \
  '
    if [ -f "{3}/raw/wh_md_walsh_degree1_results.csv" ]; then
      echo "[reuse-existing-shard] walsh_degree_1 M={1} instance_id={2}"
    else
      uv run --project "'"$PROJECT_ROOT"'" python "'"$PY_SCRIPT"'" \
        --n-sys-list '"$N_SYS"' \
        --m-values {1} \
        --instance-ids {2} \
        --num-instances-per-grid-point '"${#INSTANCE_ID_ARR[@]}"' \
        --num-restarts '"$NUM_RESTARTS"' \
        --seed-start '"$SEED_START"' \
        --steps '"$STEPS"' \
        --eval-interval '"$EVAL_INTERVAL"' \
        --learning-rate '"$LEARNING_RATE"' \
        --threshold '"$THRESHOLD"' \
        --tol '"$TOL"' \
        --su-depth '"$SU_DEPTH"' \
        --scale-init '"$SCALE_INIT"' \
        --bias-scale-init '"$BIAS_SCALE_INIT"' \
        --projection-strategy '"$PROJECTION_STRATEGY"' \
        --plot-dpi '"$PLOT_DPI"' \
        --display-floor '"$DISPLAY_FLOOR"' \
        --output-dir "{3}"
    fi
  ' :::: "$JOBLIST_PATH"

mapfile -t RESULT_CSVS < <(find "$SHARD_ROOT" -path '*/raw/wh_md_walsh_degree1_results.csv' | sort)
mapfile -t RESTART_JSONLS < <(find "$SHARD_ROOT" -path '*/raw/wh_md_walsh_degree1_restart_records.jsonl' | sort)

if [ "${#RESULT_CSVS[@]}" -eq 0 ]; then
  echo "No shard result CSVs were produced under $SHARD_ROOT" >&2
  exit 1
fi

uv run --project "$PROJECT_ROOT" python "$PY_SCRIPT" \
  --aggregate-only \
  --n-sys-list "$N_SYS" \
  --m-values "${M_LIST_ARR[@]}" \
  --instance-ids "${INSTANCE_ID_ARR[@]}" \
  --num-instances-per-grid-point "${#INSTANCE_ID_ARR[@]}" \
  --num-restarts "$NUM_RESTARTS" \
  --seed-start "$SEED_START" \
  --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --learning-rate "$LEARNING_RATE" \
  --threshold "$THRESHOLD" \
  --tol "$TOL" \
  --su-depth "$SU_DEPTH" \
  --scale-init "$SCALE_INIT" \
  --bias-scale-init "$BIAS_SCALE_INIT" \
  --projection-strategy "$PROJECTION_STRATEGY" \
  --plot-dpi "$PLOT_DPI" \
  --display-floor "$DISPLAY_FLOOR" \
  --input-result-csvs "${RESULT_CSVS[@]}" \
  --input-restart-jsonls "${RESTART_JSONLS[@]}" \
  --output-dir "$FINAL_ROOT"

echo "aggregate_root=$FINAL_ROOT"
