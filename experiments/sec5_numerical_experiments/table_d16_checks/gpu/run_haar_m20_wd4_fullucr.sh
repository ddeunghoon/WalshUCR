#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$BUNDLE_DIR/../../../.." && pwd)}"
QSD_PROJECT_ROOT="${QSD_PROJECT_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}"

export STATE_FAMILY="${STATE_FAMILY:-haar}"
export N_SYS="${N_SYS:-4}"
export M="${M:-20}"
export INSTANCE_START="${INSTANCE_START:-0}"
export INSTANCE_END="${INSTANCE_END:-9}"
export NUM_RESTARTS="${NUM_RESTARTS:-5}"
export SEED_START="${SEED_START:-0}"
export SU_DEPTH="${SU_DEPTH:-61}"
export STEPS="${STEPS:-1000}"
export MODELS="${MODELS:-walsh_degree_4 full_ucr}"
export MICROBATCH_SIZE="${MICROBATCH_SIZE:-0}"
export SCHEDULE_CHECKPOINT_CHUNK_SIZE="${SCHEDULE_CHECKPOINT_CHUNK_SIZE:-0}"
export SKIP_SDP="${SKIP_SDP:-0}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/experiments/sec5_numerical_experiments/table_d16_checks/results/memopt_haar_wd4_fullucr_nsys4_M20_instances00-09_r5_su61_steps1000_nomemopt}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QSD_PROJECT_ROOT

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
echo "models=$MODELS"
echo "state_family=$STATE_FAMILY"
echo "schedule_checkpoint_chunk_size=$SCHEDULE_CHECKPOINT_CHUNK_SIZE"
echo "microbatch_size=$MICROBATCH_SIZE"
echo "skip_sdp=$SKIP_SDP"

SDP_ARGS=()
if [ "$SKIP_SDP" = "1" ]; then
  SDP_ARGS+=("--skip-sdp")
fi
GPU_ARGS=()
if [ "${REQUIRE_GPU:-1}" = "0" ]; then
  GPU_ARGS+=("--no-require-gpu")
fi

mkdir -p "$RUN_ROOT"

RUN_COUNT=0
for INSTANCE_ID in $(seq "$INSTANCE_START" "$INSTANCE_END"); do
  for MODEL_TYPE in $MODELS; do
    INSTANCE_DIR="$(printf "%s/instance%02d/%s" "$RUN_ROOT" "$INSTANCE_ID" "$MODEL_TYPE")"
    mkdir -p "$INSTANCE_DIR"

    echo "instance_id=$INSTANCE_ID model_type=$MODEL_TYPE"
    ((RUN_COUNT += 1))
    uv run --project "$PROJECT_DIR" python "$BUNDLE_DIR/run_walsh_degree1_gpu_memopt.py" \
      --n-sys "$N_SYS" \
      --M "$M" \
      --instance-id "$INSTANCE_ID" \
      --model-type "$MODEL_TYPE" \
      --state-family "$STATE_FAMILY" \
      --num-restarts "$NUM_RESTARTS" \
      --seed-start "$SEED_START" \
      --su-depth "$SU_DEPTH" \
      --steps "$STEPS" \
      --eval-interval "${EVAL_INTERVAL:-50}" \
      --learning-rate "${LEARNING_RATE:-1e-2}" \
      --threshold "${THRESHOLD:-1e-6}" \
      --scale-init "${SCALE_INIT:-1.0}" \
      --bias-scale-init "${BIAS_SCALE_INIT:-1.0}" \
      --schedule-checkpoint-chunk-size "$SCHEDULE_CHECKPOINT_CHUNK_SIZE" \
      --microbatch-size "$MICROBATCH_SIZE" \
      "${GPU_ARGS[@]}" \
      "${SDP_ARGS[@]}" \
      --output-dir "$INSTANCE_DIR"
  done
done

echo "run_count=$RUN_COUNT"
