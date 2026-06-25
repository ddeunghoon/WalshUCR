#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

N_SYS="${N_SYS:-5}"
M="${M:-40}"
INSTANCE_START="${INSTANCE_START:-0}"
INSTANCE_END="${INSTANCE_END:-9}"
NUM_RESTARTS="${NUM_RESTARTS:-2}"
SEED_START="${SEED_START:-0}"
SU_DEPTH="${SU_DEPTH:-100}"
STEPS="${STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
THRESHOLD="${THRESHOLD:-1e-6}"
SCALE_INIT="${SCALE_INIT:-1.0}"
BIAS_SCALE_INIT="${BIAS_SCALE_INIT:-1.0}"
MODELS="${MODELS:-walsh_degree_1 full_ucr}"
STATE_FAMILY="${STATE_FAMILY:-weyl}"
SCHEDULE_CHECKPOINT_CHUNK_SIZE="${SCHEDULE_CHECKPOINT_CHUNK_SIZE:-32}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-4}"
SKIP_SDP="${SKIP_SDP:-0}"
SKIP_MISSING_EXPERIMENTS="${SKIP_MISSING_EXPERIMENTS:-0}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/experiments/sec5_numerical_experiments/table_d16_checks/results/memopt_weyl_gpu_compare_nsys5_M40_instances00-09_r2_su100_steps1000_mb${MICROBATCH_SIZE}_ckpt${SCHEDULE_CHECKPOINT_CHUNK_SIZE}}"

case "$SKIP_MISSING_EXPERIMENTS" in
  0|1)
    ;;
  *)
    echo "SKIP_MISSING_EXPERIMENTS must be 0 or 1, got '$SKIP_MISSING_EXPERIMENTS'." >&2
    exit 1
    ;;
esac

if [ "$SKIP_MISSING_EXPERIMENTS" = "1" ] && [ ! -d "$RUN_ROOT" ]; then
  echo "[skip-missing] run_root does not exist: $RUN_ROOT"
  exit 0
fi

mkdir -p "$RUN_ROOT"

echo "run_root=$RUN_ROOT"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "n_sys=$N_SYS"
echo "M=$M"
echo "instance_start=$INSTANCE_START"
echo "instance_end=$INSTANCE_END"
echo "num_restarts=$NUM_RESTARTS"
echo "seed_start=$SEED_START"
echo "su_depth=$SU_DEPTH"
echo "steps=$STEPS"
echo "eval_interval=$EVAL_INTERVAL"
echo "models=$MODELS"
echo "state_family=$STATE_FAMILY"
echo "schedule_checkpoint_chunk_size=$SCHEDULE_CHECKPOINT_CHUNK_SIZE"
echo "microbatch_size=$MICROBATCH_SIZE"
echo "skip_sdp=$SKIP_SDP"
echo "skip_missing_experiments=$SKIP_MISSING_EXPERIMENTS"

SDP_ARGS=()
if [ "$SKIP_SDP" = "1" ]; then
  SDP_ARGS+=("--skip-sdp")
fi

RUN_COUNT=0
SKIP_COUNT=0
for INSTANCE_ID in $(seq "$INSTANCE_START" "$INSTANCE_END"); do
  for MODEL_TYPE in $MODELS; do
    INSTANCE_DIR="$(printf "%s/instance%02d/%s" "$RUN_ROOT" "$INSTANCE_ID" "$MODEL_TYPE")"
    if [ "$SKIP_MISSING_EXPERIMENTS" = "1" ] && [ ! -d "$INSTANCE_DIR" ]; then
      echo "[skip-missing] instance_id=$INSTANCE_ID model_type=$MODEL_TYPE output_dir=$INSTANCE_DIR"
      ((SKIP_COUNT += 1))
      continue
    fi
    mkdir -p "$INSTANCE_DIR"

    echo "instance_id=$INSTANCE_ID model_type=$MODEL_TYPE"
    ((RUN_COUNT += 1))
    uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_walsh_degree1_gpu_memopt.py" \
      --n-sys "$N_SYS" \
      --M "$M" \
      --instance-id "$INSTANCE_ID" \
      --model-type "$MODEL_TYPE" \
      --state-family "$STATE_FAMILY" \
      --num-restarts "$NUM_RESTARTS" \
      --seed-start "$SEED_START" \
      --su-depth "$SU_DEPTH" \
      --steps "$STEPS" \
      --eval-interval "$EVAL_INTERVAL" \
      --learning-rate "$LEARNING_RATE" \
      --threshold "$THRESHOLD" \
      --scale-init "$SCALE_INIT" \
      --bias-scale-init "$BIAS_SCALE_INIT" \
      --schedule-checkpoint-chunk-size "$SCHEDULE_CHECKPOINT_CHUNK_SIZE" \
      --microbatch-size "$MICROBATCH_SIZE" \
      "${SDP_ARGS[@]}" \
      --output-dir "$INSTANCE_DIR"
  done
done

echo "run_count=$RUN_COUNT"
echo "skip_count=$SKIP_COUNT"
