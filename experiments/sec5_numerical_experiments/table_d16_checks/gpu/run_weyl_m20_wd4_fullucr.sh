#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export STATE_FAMILY="${STATE_FAMILY:-weyl}"
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
export RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/experiments/sec5_numerical_experiments/table_d16_checks/results/memopt_weyl_wd4_fullucr_nsys4_M20_instances00-09_r5_su61_steps1000_nomemopt}"

bash "$SCRIPT_DIR/run_models.sh"
