from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

CURRENT_DIR = Path(__file__).resolve().parent
SEC5_DIR = CURRENT_DIR.parent
UCR_METHOD_DIR = SEC5_DIR.parent
ROOT = SEC5_DIR.parents[2]
SRC_DIR = (SEC5_DIR / "../../../src").resolve()
if str(SEC5_DIR) not in sys.path:
    sys.path.append(str(SEC5_DIR))
if str(UCR_METHOD_DIR) not in sys.path:
    sys.path.append(str(UCR_METHOD_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from walsh_ucr.training.trainer import JAX_Debug_Trainer, JAX_Full_Trainer, TrainResult
from walsh_ucr.models.vqsd import RandomSparseFullUcrVQSD

from weyl_problem import _build_problem_instance, _parse_bool_arg
from wh_d8_sweep import (
    ModelSpec,
    _append_restart_checkpoint_record,
    _build_batched_qnode_for_problem,
    _build_model,
    _coerce_csv_value as _coerce_sec5_csv_value,
    _extract_best_objective,
    _load_restart_checkpoint_records,
    _make_restart_checkpoint_record,
    _problem_namespace,
    _resume_state_from_restart_checkpoint_records,
    _stats_payload,
    _termination_reason,
    _validate_restart_checkpoint_records,
    build_projection_groups,
    build_restart_checkpoint_path,
    compute_ucr_parameter_counts,
    make_projected_losses,
)


DEFAULT_REFERENCE_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_sweep_results.csv"
)
DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "results" / "random_sparse_ucr_vs_degree1"
DEFAULT_PLOT_DPI = 200
DEFAULT_DEVICE_NAME = "default.qubit"
DEFAULT_DIFF_METHOD = "backprop"
DEFAULT_JIT_BACKEND = "gpu"
DEFAULT_STATE_DTYPE = "complex128"
DEFAULT_SPARSE_SEED_OFFSET = 0
SPARSE_BUDGET_RULE = "blockwise_match_wd1_param_count"
SPARSE_FROZEN_FILL = "fixed_zero"
RANDOM_SPARSE_EXECUTION_ACTIVE_GATE = "active_gate"
RANDOM_SPARSE_EXECUTION_MASKED_ZERO = "masked_zero"
DEFAULT_RANDOM_SPARSE_EXECUTION = RANDOM_SPARSE_EXECUTION_ACTIVE_GATE
ACTIVE_SPARSE_FROZEN_FILL = "not_applicable_active_gate"
RESTART_CHECKPOINT_VERSION = 1

RANDOM_SPARSE_SPEC = ModelSpec(
    model_type="random_sparse_ucr",
    model_name="vqsd",
    mean_init="pi/2",
    bias_mean_init=None,
)


def normalize_random_sparse_execution(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {RANDOM_SPARSE_EXECUTION_ACTIVE_GATE, RANDOM_SPARSE_EXECUTION_MASKED_ZERO}:
        raise ValueError(
            f"Unsupported random sparse execution mode '{value}'. "
            f"Use {RANDOM_SPARSE_EXECUTION_ACTIVE_GATE} or {RANDOM_SPARSE_EXECUTION_MASKED_ZERO}."
        )
    return mode


def sparse_frozen_fill_for_execution(execution_mode: str) -> str:
    mode = normalize_random_sparse_execution(execution_mode)
    if mode == RANDOM_SPARSE_EXECUTION_ACTIVE_GATE:
        return ACTIVE_SPARSE_FROZEN_FILL
    return SPARSE_FROZEN_FILL


def infer_reference_summary_path(reference_results_csv: Path) -> Path:
    return reference_results_csv.parent.parent / "summaries" / "wh_md_sweep_summary.json"


def load_reference_summary(reference_results_csv: Path, explicit_summary_json: str | None) -> dict[str, Any]:
    if explicit_summary_json is None:
        summary_path = infer_reference_summary_path(reference_results_csv)
    else:
        summary_path = Path(explicit_summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Reference summary JSON does not exist: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["_resolved_path"] = str(summary_path)
    return payload


def build_training_args_from_reference(
    *,
    row: dict[str, Any],
    reference_summary: dict[str, Any],
    cli_args: argparse.Namespace,
) -> argparse.Namespace:
    config = dict(reference_summary.get("config", {}))

    def _resolve(config_key: str, row_key: str | None = None) -> Any:
        if config_key in config:
            return config[config_key]
        resolved_row_key = config_key if row_key is None else row_key
        if resolved_row_key in row:
            return row[resolved_row_key]
        raise KeyError(
            f"Reference config is missing required key '{config_key}' and row key "
            f"'{resolved_row_key}' is also unavailable."
        )

    return argparse.Namespace(
        n_sys=int(row["n_sys"]),
        m_outcome=int(row["M"]),
        num_restarts=int(_resolve("num_restarts")),
        steps=int(_resolve("steps", "max_steps")),
        eval_interval=int(_resolve("eval_interval")),
        optimizer=str(config.get("optimizer_name", "adam")),
        learning_rate=float(_resolve("learning_rate")),
        trainer=str(config.get("trainer", "full")),
        loss_type=str(config.get("loss_type", "linear")),
        device_name=str(cli_args.device_name),
        diff_method=str(cli_args.diff_method),
        jit_backend=str(cli_args.jit_backend),
        threshold=float(_resolve("threshold")),
        tol=float(config.get("tol", 5e-4)),
        projection_strategy=str(_resolve("projection_strategy")),
        renormalize_projected_probs=bool(config.get("renormalize_projected_probs", False)),
        state_dtype=str(cli_args.state_dtype),
        random_sparse_execution=normalize_random_sparse_execution(cli_args.random_sparse_execution),
        use_scrambler=bool(config.get("use_scrambler", True)),
        su_depth=int(config.get("su_depth", 14)),
        scale_init=float(config.get("scale_init", 0.01)),
        bias_scale_init=float(config.get("bias_scale_init", 0.01)),
        seed_start=0,
        plot_dpi=int(cli_args.plot_dpi),
        aggregate_only=bool(cli_args.aggregate_only),
    )


def build_sparse_mask_seed(*, n_sys: int, m_outcome: int, instance_id: int, sparse_seed_offset: int) -> int:
    return (
        int(sparse_seed_offset)
        + 1_000_000 * int(n_sys)
        + 10_000 * int(m_outcome)
        + int(instance_id)
    )


def build_random_sparse_ucr_mask(
    *,
    model: Any,
    n_sys: int,
    m_outcome: int,
    instance_id: int,
    sparse_seed_offset: int,
) -> tuple[jax.Array, dict[str, Any]]:
    mask_seed = build_sparse_mask_seed(
        n_sys=int(n_sys),
        m_outcome=int(m_outcome),
        instance_id=int(instance_id),
        sparse_seed_offset=int(sparse_seed_offset),
    )
    rng = np.random.default_rng(mask_seed)
    theta_dim = int(model.layout.theta_dim)
    mask = np.zeros((theta_dim,), dtype=bool)
    block_records: list[dict[str, Any]] = []

    for name in model.layout.names:
        sl, shape = model.layout.slices[name]
        if name == "SU_0" or name.startswith("MTPLX_"):
            mask[sl] = True
            continue

        if not name.startswith("UCR_"):
            raise ValueError(f"Unexpected parameter block '{name}' in full-UCR layout.")

        block_idx = int(name.split("_", maxsplit=1)[1])
        full_block_size = int(np.prod(shape))
        sparse_budget = 1 + int(n_sys) + block_idx
        if sparse_budget > full_block_size:
            raise ValueError(
                f"Sparse block budget {sparse_budget} exceeds full block size {full_block_size} for {name}."
            )
        local_indices = np.sort(
            rng.choice(full_block_size, size=sparse_budget, replace=False).astype(np.int32)
        )
        mask[sl.start + local_indices] = True
        block_records.append(
            {
                "block_name": name,
                "block_index": int(block_idx),
                "full_block_size": int(full_block_size),
                "sparse_block_budget": int(sparse_budget),
                "selected_local_indices": [int(value) for value in local_indices.tolist()],
            }
        )

    num_ucr_params_sparse = int(sum(record["sparse_block_budget"] for record in block_records))
    num_ucr_params_full, num_ucr_params_degree1_budget, compression_ratio = compute_ucr_parameter_counts(
        n_sys=int(n_sys),
        n_anc=int(model.n_anc),
    )
    if num_ucr_params_sparse != num_ucr_params_degree1_budget:
        raise ValueError(
            "Random sparse UCR budget must match the WD-1 UCR parameter count exactly, got "
            f"sparse={num_ucr_params_sparse}, wd1_budget={num_ucr_params_degree1_budget}."
        )

    payload = {
        "checkpoint_version": int(RESTART_CHECKPOINT_VERSION),
        "model_type": str(RANDOM_SPARSE_SPEC.model_type),
        "model_name": str(RANDOM_SPARSE_SPEC.model_name),
        "n_sys": int(n_sys),
        "d": int(2 ** int(n_sys)),
        "M": int(m_outcome),
        "instance_id": int(instance_id),
        "mask_seed": int(mask_seed),
        "sparse_seed_offset": int(sparse_seed_offset),
        "sparse_budget_rule": SPARSE_BUDGET_RULE,
        "sparse_frozen_fill": SPARSE_FROZEN_FILL,
        "num_ucr_params_sparse": int(num_ucr_params_sparse),
        "num_ucr_params_degree1_budget": int(num_ucr_params_degree1_budget),
        "num_ucr_params_full": int(num_ucr_params_full),
        "ucr_compression_ratio_full_to_sparse": float(compression_ratio),
        "trainable_param_count_total": int(np.count_nonzero(mask)),
        "theta_dim_total": int(theta_dim),
        "selected_ucr_blocks": block_records,
    }
    return jnp.asarray(mask, dtype=bool), payload


def _selected_ucr_indices_from_mask_payload(
    mask_payload: dict[str, Any],
    *,
    n_anc: int,
) -> tuple[tuple[int, ...], ...]:
    records = sorted(
        mask_payload.get("selected_ucr_blocks", []),
        key=lambda item: int(item["block_index"]),
    )
    if len(records) != int(n_anc):
        raise ValueError(f"Expected {n_anc} selected UCR block records, got {len(records)}.")
    return tuple(
        tuple(int(value) for value in record["selected_local_indices"])
        for record in records
    )


def _build_active_random_sparse_model_and_payload(
    *,
    n_sys: int,
    n_anc: int,
    m_outcome: int,
    instance_id: int,
    sparse_seed_offset: int,
    su_depth: int,
    scale_init: float,
    bias_scale_init: float,
) -> tuple[RandomSparseFullUcrVQSD, dict[str, Any]]:
    full_reference_model = _build_model(
        RANDOM_SPARSE_SPEC,
        n_sys=int(n_sys),
        n_anc=int(n_anc),
        su_depth=int(su_depth),
        scale_init=float(scale_init),
        bias_scale_init=float(bias_scale_init),
    )
    _, full_mask_payload = build_random_sparse_ucr_mask(
        model=full_reference_model,
        n_sys=int(n_sys),
        m_outcome=int(m_outcome),
        instance_id=int(instance_id),
        sparse_seed_offset=int(sparse_seed_offset),
    )
    selected_indices = _selected_ucr_indices_from_mask_payload(
        full_mask_payload,
        n_anc=int(n_anc),
    )
    model = RandomSparseFullUcrVQSD(
        n_anc=int(n_anc),
        n_sys=int(n_sys),
        selected_ucr_indices=selected_indices,
        su_depth=int(su_depth),
        mean_init=str(RANDOM_SPARSE_SPEC.mean_init),
        scale_init=float(scale_init),
    )

    payload = dict(full_mask_payload)
    full_theta_dim = int(payload["theta_dim_total"])
    payload.update(
        {
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION_ACTIVE_GATE,
            "sparse_frozen_fill": ACTIVE_SPARSE_FROZEN_FILL,
            "theta_dim_full_reference": int(full_theta_dim),
            "theta_dim_total": int(model.layout.theta_dim),
            "trainable_param_count_total": int(model.layout.theta_dim),
            "active_gate_ucr_branch_count": int(payload["num_ucr_params_sparse"]),
            "omitted_zero_ucr_branch_count": int(
                payload["num_ucr_params_full"] - payload["num_ucr_params_sparse"]
            ),
        }
    )
    return model, payload


def _record_execution_mode(record: dict[str, Any]) -> str:
    return normalize_random_sparse_execution(
        record.get("random_sparse_execution", RANDOM_SPARSE_EXECUTION_MASKED_ZERO)
    )


def _checkpoint_records_for_execution(
    checkpoint_path: Path | None,
    *,
    execution_mode: str,
    theta_dim: int | None = None,
) -> list[dict[str, Any]]:
    if checkpoint_path is None:
        return []
    mode = normalize_random_sparse_execution(execution_mode)
    records: list[dict[str, Any]] = []
    for record in _load_restart_checkpoint_records(checkpoint_path):
        if _record_execution_mode(record) != mode:
            continue
        if theta_dim is not None:
            theta = record.get("theta", [])
            if len(theta) != int(theta_dim):
                raise ValueError(
                    f"{mode} checkpoint has incompatible theta length: "
                    f"expected {theta_dim}, got {len(theta)}."
                )
        records.append(record)
    return records


def apply_parameter_mask(theta_raw: jax.Array, trainable_mask: jax.Array, fixed_theta: jax.Array) -> jax.Array:
    theta_arr = jnp.asarray(theta_raw, dtype=jnp.float64)
    return jnp.where(trainable_mask, theta_arr, fixed_theta)


def make_masked_loss_fns(
    *,
    train_loss_fn: Any,
    eval_loss_fn: Any,
    trainable_mask: jax.Array,
    fixed_theta: jax.Array,
) -> tuple[Any, Any]:
    def masked_train_loss(theta_raw: jax.Array, *loss_args: Any) -> jax.Array:
        theta_eff = apply_parameter_mask(theta_raw, trainable_mask, fixed_theta)
        return train_loss_fn(theta_eff, *loss_args)

    def masked_eval_loss(theta_raw: jax.Array, *loss_args: Any) -> Any:
        theta_eff = apply_parameter_mask(theta_raw, trainable_mask, fixed_theta)
        return eval_loss_fn(theta_eff, *loss_args)

    return masked_train_loss, masked_eval_loss


def _make_shared_full_trainer(
    *,
    train_loss_fn: Any,
    eval_loss_fn: Any,
    theta_template: Any,
    m_outcome: int,
    learning_rate: float,
    eval_interval: int,
    jit_backend: str | None,
) -> JAX_Full_Trainer:
    a_priori_probs = jnp.ones((int(m_outcome),), dtype=jnp.float64) / float(m_outcome)
    return JAX_Full_Trainer(
        train_cost_fn=train_loss_fn,
        theta_init=theta_template,
        optimizer_name="adam",
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        memory_size=10,
        eval_interval=int(eval_interval),
        eval_cost_fn=eval_loss_fn,
        n_outcome=int(m_outcome),
        a_priori_probs=a_priori_probs,
        jit_backend=jit_backend,
    )


def _run_masked_random_sparse_restarts(
    *,
    problem: dict[str, Any],
    args: argparse.Namespace,
    target_states: jax.Array,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    sparse_seed_offset: int,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    n_sys = int(args.n_sys)
    n_anc = int(problem["n_anc"])
    model = _build_model(
        RANDOM_SPARSE_SPEC,
        n_sys=n_sys,
        n_anc=n_anc,
        su_depth=int(args.su_depth),
        scale_init=float(args.scale_init),
        bias_scale_init=float(args.bias_scale_init),
    )
    mask, mask_payload = build_random_sparse_ucr_mask(
        model=model,
        n_sys=n_sys,
        m_outcome=int(args.m_outcome),
        instance_id=int(instance_id),
        sparse_seed_offset=int(sparse_seed_offset),
    )
    mask_payload.update(
        {
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION_MASKED_ZERO,
            "theta_dim_full_reference": int(model.layout.theta_dim),
        }
    )
    fixed_theta = jnp.zeros((int(model.layout.theta_dim),), dtype=jnp.float64)

    raw_outcomes = 2 ** int(problem["n_anc"])
    groups, _ = build_projection_groups(
        raw_outcomes=raw_outcomes,
        m_outcomes=int(args.m_outcome),
        strategy=str(args.projection_strategy),
    )
    batched_qnode = _build_batched_qnode_for_problem(
        problem=problem,
        model=model,
        n_sys=n_sys,
        device_name=str(args.device_name),
        diff_method=str(args.diff_method),
    )
    train_loss_fn, eval_loss_fn, _ = make_projected_losses(
        batched_qnode,
        groups=groups,
        loss_type=str(args.loss_type),
        renormalize_projected_probs=bool(args.renormalize_projected_probs),
    )
    masked_train_loss_fn, masked_eval_loss_fn = make_masked_loss_fns(
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        trainable_mask=mask,
        fixed_theta=fixed_theta,
    )
    a_priori_probs = jnp.ones((int(args.m_outcome),), dtype=jnp.float64) / float(args.m_outcome)
    use_restart_reuse = (
        str(args.trainer) == "full" and str(args.optimizer).lower() == "adam"
    )
    if use_restart_reuse:
        theta_template = model.layout.init_params(jax.random.PRNGKey(int(args.seed_start)))
        shared_trainer = _make_shared_full_trainer(
            train_loss_fn=masked_train_loss_fn,
            eval_loss_fn=masked_eval_loss_fn,
            theta_template=theta_template,
            m_outcome=int(args.m_outcome),
            learning_rate=float(args.learning_rate),
            eval_interval=int(args.eval_interval),
            jit_backend=getattr(args, "jit_backend", None),
        )
        train_args = (problem["inputs"], target_states)
        eval_args = (problem["inputs"], target_states)
    else:
        trainer_cls = JAX_Full_Trainer if str(args.trainer) == "full" else JAX_Debug_Trainer

    best_record: dict[str, Any] | None = None
    best_theta_eff: jax.Array | None = None
    restart_records: list[dict[str, Any]] = []
    completed_restart_ids: set[int] = set()

    if checkpoint_path is not None:
        checkpoint_records = [
            record
            for record in _checkpoint_records_for_execution(
                checkpoint_path,
                execution_mode=RANDOM_SPARSE_EXECUTION_MASKED_ZERO,
                theta_dim=int(model.layout.theta_dim),
            )
            if int(record["restart_id"]) < int(args.num_restarts)
        ]
        if checkpoint_records:
            _validate_restart_checkpoint_records(
                checkpoint_records,
                spec=RANDOM_SPARSE_SPEC,
                n_sys=int(args.n_sys),
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            restart_records, best_record, best_theta_eff, completed_restart_ids = (
                _resume_state_from_restart_checkpoint_records(checkpoint_records)
            )
            for row in restart_records:
                row["random_sparse_execution"] = RANDOM_SPARSE_EXECUTION_MASKED_ZERO
            print(
                f"[resume][{RANDOM_SPARSE_EXECUTION_MASKED_ZERO}] "
                f"model={RANDOM_SPARSE_SPEC.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} completed_restarts={len(completed_restart_ids)}",
                flush=True,
            )

    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_restart_ids:
            continue
        seed_opt = int(args.seed_start) + restart_id
        theta_init = model.layout.init_params(jax.random.PRNGKey(seed_opt))
        if use_restart_reuse:
            opt_state0 = shared_trainer.optimizer.init(theta_init)
            start_time = time.perf_counter()
            result_dict = shared_trainer._solve_adam(
                theta_init,
                opt_state0,
                train_args,
                eval_args,
                max_steps=int(args.steps),
                threshold=float(args.threshold),
                eval_interval=int(args.eval_interval),
                early_stop=True,
                train_tolerance=1e-4,
                switch_step=-1,
            )
            jax.block_until_ready(result_dict["theta"])
            wall_clock_sec = time.perf_counter() - start_time
            result = TrainResult(
                theta=result_dict["theta"],
                steps_run=int(result_dict["steps_run"]),
                last_eval_loss=float(result_dict["last_eval_loss"]),
                stopped_early=bool(result_dict["stopped_early"]),
                nan_found=bool(result_dict["nan_found"]),
                step_log=result_dict["step_log"],
                loss_log=result_dict["loss_log"],
            )
        else:
            trainer = trainer_cls(
                train_cost_fn=masked_train_loss_fn,
                theta_init=theta_init,
                optimizer_name="adam",
                learning_rate=float(args.learning_rate),
                weight_decay=0.0,
                memory_size=10,
                eval_interval=int(args.eval_interval),
                jit_backend=str(args.jit_backend),
                eval_cost_fn=masked_eval_loss_fn,
                n_outcome=int(args.m_outcome),
                a_priori_probs=a_priori_probs,
            )
            start_time = time.perf_counter()
            result = trainer.run_optimization(
                steps=int(args.steps),
                train_args=(problem["inputs"], target_states),
                eval_args=(problem["inputs"], target_states),
                threshold=float(args.threshold),
                verbose=0,
                eval_interval=int(args.eval_interval),
            )
            wall_clock_sec = time.perf_counter() - start_time

        final_objective = float(result.last_eval_loss)
        best_objective = _extract_best_objective(result.loss_log)
        p_succ = float(1.0 - final_objective)
        theta_eff = apply_parameter_mask(result.theta, mask, fixed_theta)
        record = {
            "model_type": RANDOM_SPARSE_SPEC.model_type,
            "model_name": RANDOM_SPARSE_SPEC.model_name,
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION_MASKED_ZERO,
            "seed_opt": int(seed_opt),
            "restart_id": int(restart_id),
            "num_steps": int(result.steps_run),
            "termination_reason": _termination_reason(result, max_steps=int(args.steps)),
            "best_objective_value": float(best_objective),
            "final_objective_value": float(final_objective),
            "p_succ": float(p_succ),
            "wall_clock_sec": float(wall_clock_sec),
        }
        restart_records.append(record)
        if checkpoint_path is not None:
            checkpoint_record = _make_restart_checkpoint_record(
                record=record,
                theta=theta_eff,
                spec=RANDOM_SPARSE_SPEC,
                n_sys=int(args.n_sys),
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            checkpoint_record.update(
                {
                    "random_sparse_execution": RANDOM_SPARSE_EXECUTION_MASKED_ZERO,
                    "sparse_frozen_fill": SPARSE_FROZEN_FILL,
                    "theta_dim": int(model.layout.theta_dim),
                    "theta_dim_full_reference": int(model.layout.theta_dim),
                }
            )
            _append_restart_checkpoint_record(checkpoint_path, checkpoint_record)
            print(
                f"[restart][checkpointed][{RANDOM_SPARSE_EXECUTION_MASKED_ZERO}]"
                f"{'[restart_reuse]' if use_restart_reuse else ''} "
                f"model={RANDOM_SPARSE_SPEC.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} restart_id={restart_id}",
                flush=True,
            )

        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = record
            best_theta_eff = theta_eff

    if best_record is None or best_theta_eff is None:
        raise RuntimeError("No valid restart result found for random sparse UCR.")

    summary = {
        "model_type": RANDOM_SPARSE_SPEC.model_type,
        "model_name": RANDOM_SPARSE_SPEC.model_name,
        "random_sparse_execution": RANDOM_SPARSE_EXECUTION_MASKED_ZERO,
        "p_succ": float(best_record["p_succ"]),
        "best_restart": int(best_record["restart_id"]),
        "seed_opt": int(best_record["seed_opt"]),
        "num_steps": int(best_record["num_steps"]),
        "termination_reason": str(best_record["termination_reason"]),
        "best_objective_value": float(best_record["best_objective_value"]),
        "final_objective_value": float(best_record["final_objective_value"]),
        "wall_clock_sec": float(best_record["wall_clock_sec"]),
        "theta": best_theta_eff,
    }
    return summary, restart_records, mask_payload


def _run_active_random_sparse_restarts(
    *,
    problem: dict[str, Any],
    args: argparse.Namespace,
    target_states: jax.Array,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    sparse_seed_offset: int,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    n_sys = int(args.n_sys)
    n_anc = int(problem["n_anc"])
    model, mask_payload = _build_active_random_sparse_model_and_payload(
        n_sys=n_sys,
        n_anc=n_anc,
        m_outcome=int(args.m_outcome),
        instance_id=int(instance_id),
        sparse_seed_offset=int(sparse_seed_offset),
        su_depth=int(args.su_depth),
        scale_init=float(args.scale_init),
        bias_scale_init=float(args.bias_scale_init),
    )

    raw_outcomes = 2 ** int(problem["n_anc"])
    groups, _ = build_projection_groups(
        raw_outcomes=raw_outcomes,
        m_outcomes=int(args.m_outcome),
        strategy=str(args.projection_strategy),
    )
    batched_qnode = _build_batched_qnode_for_problem(
        problem=problem,
        model=model,
        n_sys=n_sys,
        device_name=str(args.device_name),
        diff_method=str(args.diff_method),
    )
    train_loss_fn, eval_loss_fn, _ = make_projected_losses(
        batched_qnode,
        groups=groups,
        loss_type=str(args.loss_type),
        renormalize_projected_probs=bool(args.renormalize_projected_probs),
    )
    a_priori_probs = jnp.ones((int(args.m_outcome),), dtype=jnp.float64) / float(args.m_outcome)
    use_restart_reuse = str(args.trainer) == "full" and str(args.optimizer).lower() == "adam"
    if use_restart_reuse:
        theta_template = model.layout.init_params(jax.random.PRNGKey(int(args.seed_start)))
        shared_trainer = _make_shared_full_trainer(
            train_loss_fn=train_loss_fn,
            eval_loss_fn=eval_loss_fn,
            theta_template=theta_template,
            m_outcome=int(args.m_outcome),
            learning_rate=float(args.learning_rate),
            eval_interval=int(args.eval_interval),
            jit_backend=getattr(args, "jit_backend", None),
        )
        train_args = (problem["inputs"], target_states)
        eval_args = (problem["inputs"], target_states)
    else:
        trainer_cls = JAX_Full_Trainer if str(args.trainer) == "full" else JAX_Debug_Trainer

    best_record: dict[str, Any] | None = None
    best_theta: jax.Array | None = None
    restart_records: list[dict[str, Any]] = []
    completed_restart_ids: set[int] = set()

    if checkpoint_path is not None:
        checkpoint_records = [
            record
            for record in _checkpoint_records_for_execution(
                checkpoint_path,
                execution_mode=RANDOM_SPARSE_EXECUTION_ACTIVE_GATE,
                theta_dim=int(model.layout.theta_dim),
            )
            if int(record["restart_id"]) < int(args.num_restarts)
        ]
        if checkpoint_records:
            _validate_restart_checkpoint_records(
                checkpoint_records,
                spec=RANDOM_SPARSE_SPEC,
                n_sys=int(args.n_sys),
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            restart_records, best_record, best_theta, completed_restart_ids = (
                _resume_state_from_restart_checkpoint_records(checkpoint_records)
            )
            for row in restart_records:
                row["random_sparse_execution"] = RANDOM_SPARSE_EXECUTION_ACTIVE_GATE
            print(
                f"[resume][{RANDOM_SPARSE_EXECUTION_ACTIVE_GATE}] "
                f"model={RANDOM_SPARSE_SPEC.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} "
                f"completed_restarts={len(completed_restart_ids)}",
                flush=True,
            )

    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_restart_ids:
            continue
        seed_opt = int(args.seed_start) + restart_id
        theta_init = model.layout.init_params(jax.random.PRNGKey(seed_opt))
        if use_restart_reuse:
            opt_state0 = shared_trainer.optimizer.init(theta_init)
            start_time = time.perf_counter()
            result_dict = shared_trainer._solve_adam(
                theta_init,
                opt_state0,
                train_args,
                eval_args,
                max_steps=int(args.steps),
                threshold=float(args.threshold),
                eval_interval=int(args.eval_interval),
                early_stop=True,
                train_tolerance=1e-4,
                switch_step=-1,
            )
            jax.block_until_ready(result_dict["theta"])
            wall_clock_sec = time.perf_counter() - start_time
            result = TrainResult(
                theta=result_dict["theta"],
                steps_run=int(result_dict["steps_run"]),
                last_eval_loss=float(result_dict["last_eval_loss"]),
                stopped_early=bool(result_dict["stopped_early"]),
                nan_found=bool(result_dict["nan_found"]),
                step_log=result_dict["step_log"],
                loss_log=result_dict["loss_log"],
            )
        else:
            trainer = trainer_cls(
                train_cost_fn=train_loss_fn,
                theta_init=theta_init,
                optimizer_name="adam",
                learning_rate=float(args.learning_rate),
                weight_decay=0.0,
                memory_size=10,
                eval_interval=int(args.eval_interval),
                jit_backend=str(args.jit_backend),
                eval_cost_fn=eval_loss_fn,
                n_outcome=int(args.m_outcome),
                a_priori_probs=a_priori_probs,
            )
            start_time = time.perf_counter()
            result = trainer.run_optimization(
                steps=int(args.steps),
                train_args=(problem["inputs"], target_states),
                eval_args=(problem["inputs"], target_states),
                threshold=float(args.threshold),
                verbose=0,
                eval_interval=int(args.eval_interval),
            )
            wall_clock_sec = time.perf_counter() - start_time

        final_objective = float(result.last_eval_loss)
        best_objective = _extract_best_objective(result.loss_log)
        p_succ = float(1.0 - final_objective)
        record = {
            "model_type": RANDOM_SPARSE_SPEC.model_type,
            "model_name": RANDOM_SPARSE_SPEC.model_name,
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION_ACTIVE_GATE,
            "seed_opt": int(seed_opt),
            "restart_id": int(restart_id),
            "num_steps": int(result.steps_run),
            "termination_reason": _termination_reason(result, max_steps=int(args.steps)),
            "best_objective_value": float(best_objective),
            "final_objective_value": float(final_objective),
            "p_succ": float(p_succ),
            "wall_clock_sec": float(wall_clock_sec),
        }
        restart_records.append(record)
        if checkpoint_path is not None:
            checkpoint_record = _make_restart_checkpoint_record(
                record=record,
                theta=result.theta,
                spec=RANDOM_SPARSE_SPEC,
                n_sys=int(args.n_sys),
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            checkpoint_record.update(
                {
                    "random_sparse_execution": RANDOM_SPARSE_EXECUTION_ACTIVE_GATE,
                    "sparse_frozen_fill": ACTIVE_SPARSE_FROZEN_FILL,
                    "theta_dim": int(model.layout.theta_dim),
                    "theta_dim_full_reference": int(mask_payload["theta_dim_full_reference"]),
                }
            )
            _append_restart_checkpoint_record(checkpoint_path, checkpoint_record)
            print(
                f"[restart][checkpointed][{RANDOM_SPARSE_EXECUTION_ACTIVE_GATE}]"
                f"{'[restart_reuse]' if use_restart_reuse else ''} "
                f"model={RANDOM_SPARSE_SPEC.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} restart_id={restart_id}",
                flush=True,
            )

        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = record
            best_theta = result.theta

    if best_record is None or best_theta is None:
        raise RuntimeError("No valid restart result found for active random sparse UCR.")

    summary = {
        "model_type": RANDOM_SPARSE_SPEC.model_type,
        "model_name": RANDOM_SPARSE_SPEC.model_name,
        "random_sparse_execution": RANDOM_SPARSE_EXECUTION_ACTIVE_GATE,
        "p_succ": float(best_record["p_succ"]),
        "best_restart": int(best_record["restart_id"]),
        "seed_opt": int(best_record["seed_opt"]),
        "num_steps": int(best_record["num_steps"]),
        "termination_reason": str(best_record["termination_reason"]),
        "best_objective_value": float(best_record["best_objective_value"]),
        "final_objective_value": float(best_record["final_objective_value"]),
        "wall_clock_sec": float(best_record["wall_clock_sec"]),
        "theta": best_theta,
    }
    return summary, restart_records, mask_payload


def _run_random_sparse_restarts(
    *,
    problem: dict[str, Any],
    args: argparse.Namespace,
    target_states: jax.Array,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    sparse_seed_offset: int,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    execution_mode = normalize_random_sparse_execution(
        getattr(args, "random_sparse_execution", DEFAULT_RANDOM_SPARSE_EXECUTION)
    )
    if execution_mode == RANDOM_SPARSE_EXECUTION_ACTIVE_GATE:
        return _run_active_random_sparse_restarts(
            problem=problem,
            args=args,
            target_states=target_states,
            instance_id=instance_id,
            benchmark_seed=benchmark_seed,
            data_seed=data_seed,
            sparse_seed_offset=sparse_seed_offset,
            checkpoint_path=checkpoint_path,
        )
    return _run_masked_random_sparse_restarts(
        problem=problem,
        args=args,
        target_states=target_states,
        instance_id=instance_id,
        benchmark_seed=benchmark_seed,
        data_seed=data_seed,
        sparse_seed_offset=sparse_seed_offset,
        checkpoint_path=checkpoint_path,
    )


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["d"]), int(row["M"]))
        grouped.setdefault(key, []).append(row)

    metrics = (
        "gap_abs_full_ref",
        "gap_abs_random_sparse",
    )

    aggregated_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        first = bucket[0]
        payload = {
            "n_sys": int(first["n_sys"]),
            "d": int(first["d"]),
            "M": int(first["M"]),
            "M_over_d": float(first["M_over_d"]),
            "count": int(len(bucket)),
        }
        for metric_name in metrics:
            stats = _stats_payload([float(row[metric_name]) for row in bucket])
            for stat_name, stat_value in stats.items():
                payload[f"{metric_name}_{stat_name}"] = stat_value
        aggregated_rows.append(payload)

    regime_summary: dict[str, Any] = {}
    for regime_name, predicate in (
        ("M_over_d_le_1", lambda row: float(row["M_over_d"]) <= 1.0),
        ("M_over_d_gt_1", lambda row: float(row["M_over_d"]) > 1.0),
    ):
        bucket = [row for row in rows if predicate(row)]
        regime_payload = {"count": int(len(bucket))}
        for metric_name in metrics:
            regime_payload[metric_name] = _stats_payload([float(row[metric_name]) for row in bucket])
        regime_summary[regime_name] = regime_payload

    return aggregated_rows, regime_summary


def _plot_gap(aggregated_rows: Sequence[dict[str, Any]], path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = {
        "full_ucr_ref": {"color": "tab:blue", "marker": "o"},
        "random_sparse_ucr": {"color": "tab:green", "marker": "^"},
    }
    line_styles = {3: "-", 4: "--", 5: "-."}

    for n_sys in sorted({int(row["n_sys"]) for row in aggregated_rows}):
        rows_n = sorted(
            [row for row in aggregated_rows if int(row["n_sys"]) == n_sys],
            key=lambda row: float(row["M_over_d"]),
        )
        x = np.asarray([float(row["M_over_d"]) for row in rows_n], dtype=np.float64)
        d = int(rows_n[0]["d"])
        for metric_name, label_key in (("gap_abs_full_ref", "full_ucr_ref"), ("gap_abs_random_sparse", "random_sparse_ucr")):
            y = np.asarray([float(row[f"{metric_name}_mean"]) for row in rows_n], dtype=np.float64)
            yerr = np.asarray([float(row[f"{metric_name}_se"]) for row in rows_n], dtype=np.float64)
            style = styles[label_key]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=style["color"],
                marker=style["marker"],
                linestyle=line_styles.get(n_sys, ":"),
                linewidth=1.6,
                markersize=5.5,
                capsize=3.0,
                label=f"{label_key} (d={d})",
            )

    ax.set_xlabel("M/d")
    ax.set_ylabel("Absolute optimum gap")
    ax.set_title("Random sparse UCR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _results_fieldnames() -> list[str]:
    return [
        "instance_id",
        "n_sys",
        "d",
        "M",
        "M_over_d",
        "fiducial_id",
        "orbit_index_set_id",
        "prior_type",
        "benchmark_seed",
        "data_seed",
        "raw_outcomes",
        "effective_m_outcomes",
        "projection_strategy",
        "class_group_sizes",
        "coverage_ratio",
        "p_opt",
        "p_opt_ref",
        "p_succ_full",
        "p_succ_full_ref",
        "gap_abs_full",
        "gap_abs_full_ref",
        "gap_rel_full",
        "gap_rel_full_ref",
        "p_succ_random_sparse",
        "gap_abs_random_sparse",
        "gap_rel_random_sparse",
        "num_ucr_params_full",
        "num_ucr_params_degree1_budget",
        "num_ucr_params_sparse",
        "ucr_compression_ratio",
        "sparse_budget_rule",
        "sparse_frozen_fill",
        "random_sparse_execution",
        "theta_dim_total",
        "theta_dim_full_reference",
        "trainable_param_count_total",
        "active_gate_ucr_branch_count",
        "omitted_zero_ucr_branch_count",
        "mask_seed",
        "optimizer_name",
        "learning_rate",
        "learning_rate_schedule",
        "max_steps",
        "eval_interval",
        "threshold",
        "num_restarts",
        "stopping_rule",
        "aggregation_rule",
        "best_restart_full",
        "best_restart_random_sparse",
        "num_steps_full",
        "num_steps_random_sparse",
        "termination_reason_full",
        "termination_reason_random_sparse",
        "wall_clock_sec_full",
        "wall_clock_sec_random_sparse",
    ]


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = _results_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "figures": figures_dir, "summaries": summaries_dir}


def _coerce_csv_value(key: str, value: str) -> Any:
    if value == "":
        return value
    if key in {
        "p_opt_ref",
        "p_succ_full_ref",
        "gap_abs_full_ref",
        "gap_rel_full_ref",
        "p_succ_random_sparse",
        "gap_abs_random_sparse",
        "gap_rel_random_sparse",
        "wall_clock_sec_random_sparse",
    }:
        return float(value)
    if key in {
        "num_ucr_params_sparse",
        "theta_dim_total",
        "theta_dim_full_reference",
        "trainable_param_count_total",
        "active_gate_ucr_branch_count",
        "omitted_zero_ucr_branch_count",
        "mask_seed",
        "best_restart_random_sparse",
        "num_steps_random_sparse",
    }:
        return int(value)
    return _coerce_sec5_csv_value(key, value)


def _load_rows_from_csvs(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                rows.append({key: _coerce_csv_value(key, value) for key, value in raw_row.items()})
    return rows


def _load_jsonl_rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def select_reference_rows(
    rows: Sequence[dict[str, Any]],
    *,
    m_values: Sequence[int] | None,
    instance_ids: Sequence[int] | None,
) -> list[dict[str, Any]]:
    selected = list(rows)
    if m_values:
        m_filter = {int(value) for value in m_values}
        selected = [row for row in selected if int(row["M"]) in m_filter]
    if instance_ids:
        instance_filter = {int(value) for value in instance_ids}
        selected = [row for row in selected if int(row["instance_id"]) in instance_filter]
    selected.sort(key=lambda row: (int(row["n_sys"]), int(row["M"]), int(row["instance_id"])))
    return selected


def _materialize_outputs(
    *,
    rows: Sequence[dict[str, Any]],
    restart_rows: Sequence[dict[str, Any]],
    mask_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    reference_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    dirs = _prepare_output_dirs(output_dir)
    aggregated_rows, regime_summary = aggregate_rows(rows)

    raw_csv_path = dirs["raw"] / "random_sparse_vs_degree1_results.csv"
    restart_jsonl_path = dirs["raw"] / "random_sparse_restart_records.jsonl"
    mask_jsonl_path = dirs["raw"] / "random_sparse_mask_records.jsonl"
    gap_plot_path = dirs["figures"] / "random_sparse_gap_plot.png"
    summary_json_path = dirs["summaries"] / "random_sparse_vs_degree1_summary.json"

    _write_results_csv(raw_csv_path, rows)
    _write_jsonl(restart_jsonl_path, restart_rows)
    _write_jsonl(mask_jsonl_path, mask_rows)
    _plot_gap(aggregated_rows, gap_plot_path, dpi=int(args.plot_dpi))

    summary = {
        "config": {
            "reference_results_csv": str(Path(args.reference_results_csv).expanduser().resolve()),
            "reference_summary_json": (
                str(reference_summary["_resolved_path"]) if reference_summary is not None else None
            ),
            "selected_m_values": [int(value) for value in args.m_values] if args.m_values else None,
            "selected_instance_ids": (
                [int(value) for value in args.instance_ids] if args.instance_ids else None
            ),
            "sparse_seed_offset": int(args.sparse_seed_offset),
            "sparse_budget_rule": SPARSE_BUDGET_RULE,
            "sparse_frozen_fill": sparse_frozen_fill_for_execution(args.random_sparse_execution),
            "random_sparse_execution": normalize_random_sparse_execution(args.random_sparse_execution),
            "aggregate_only": bool(args.aggregate_only),
            "device_name": str(args.device_name),
            "diff_method": str(args.diff_method),
            "jit_backend": str(args.jit_backend),
            "state_dtype": str(args.state_dtype),
            "plot_dpi": int(args.plot_dpi),
            "reference_experiment_config": (
                dict(reference_summary.get("config", {})) if reference_summary is not None else None
            ),
        },
        "counts": {
            "num_rows": int(len(rows)),
            "num_restart_rows": int(len(restart_rows)),
            "num_mask_rows": int(len(mask_rows)),
        },
        "aggregated_by_grid": aggregated_rows,
        "regime_summary": regime_summary,
        "artifacts": {
            "output_dir": str(output_dir),
            "results_csv": str(raw_csv_path),
            "restart_records_jsonl": str(restart_jsonl_path),
            "mask_records_jsonl": str(mask_jsonl_path),
            "gap_plot_png": str(gap_plot_path),
            "summary_json": str(summary_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {raw_csv_path}")
    print(f"saved: {restart_jsonl_path}")
    print(f"saved: {mask_jsonl_path}")
    print(f"saved: {gap_plot_path}")
    print(f"saved: {summary_json_path}")
    return summary


def aggregate_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input_result_csvs:
        raise ValueError("--aggregate-only requires --input-result-csvs.")
    rows = _load_rows_from_csvs(args.input_result_csvs)
    restart_rows = _load_jsonl_rows(args.input_restart_jsonls)
    mask_rows = _load_jsonl_rows(args.input_mask_jsonls)
    reference_summary = None
    reference_csv_path = Path(args.reference_results_csv).expanduser().resolve()
    if reference_csv_path.exists():
        reference_summary = load_reference_summary(reference_csv_path, args.reference_summary_json)
    output_dir = Path(args.output_dir).expanduser().resolve()
    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        mask_rows=mask_rows,
        args=args,
        output_dir=output_dir,
        reference_summary=reference_summary,
    )


def build_result_row(
    *,
    reference_row: dict[str, Any],
    sparse_summary: dict[str, Any],
    mask_payload: dict[str, Any],
    run_args: argparse.Namespace,
) -> dict[str, Any]:
    row = dict(reference_row)
    p_opt = float(reference_row["p_opt"])
    p_succ_random_sparse = float(sparse_summary["p_succ"])
    gap_abs_random_sparse = float(p_opt - p_succ_random_sparse)

    row.update(
        {
            "p_opt_ref": float(reference_row["p_opt"]),
            "p_succ_full_ref": float(reference_row["p_succ_full"]),
            "gap_abs_full_ref": float(reference_row["gap_abs_full"]),
            "gap_rel_full_ref": float(reference_row["gap_rel_full"]),
            "p_succ_random_sparse": float(p_succ_random_sparse),
            "gap_abs_random_sparse": float(gap_abs_random_sparse),
            "gap_rel_random_sparse": float(gap_abs_random_sparse / max(p_opt, 1e-12)),
            "num_ucr_params_sparse": int(mask_payload["num_ucr_params_sparse"]),
            "num_ucr_params_degree1_budget": int(mask_payload["num_ucr_params_degree1_budget"]),
            "sparse_budget_rule": str(mask_payload["sparse_budget_rule"]),
            "sparse_frozen_fill": str(mask_payload["sparse_frozen_fill"]),
            "random_sparse_execution": str(mask_payload.get("random_sparse_execution", run_args.random_sparse_execution)),
            "theta_dim_total": int(mask_payload.get("theta_dim_total", 0)),
            "theta_dim_full_reference": int(mask_payload.get("theta_dim_full_reference", mask_payload.get("theta_dim_total", 0))),
            "trainable_param_count_total": int(mask_payload.get("trainable_param_count_total", 0)),
            "active_gate_ucr_branch_count": int(mask_payload.get("active_gate_ucr_branch_count", 0)),
            "omitted_zero_ucr_branch_count": int(mask_payload.get("omitted_zero_ucr_branch_count", 0)),
            "mask_seed": int(mask_payload["mask_seed"]),
            "optimizer_name": str(run_args.optimizer),
            "learning_rate": float(run_args.learning_rate),
            "learning_rate_schedule": "constant",
            "max_steps": int(run_args.steps),
            "eval_interval": int(run_args.eval_interval),
            "threshold": float(run_args.threshold),
            "num_restarts": int(run_args.num_restarts),
            "best_restart_random_sparse": int(sparse_summary["best_restart"]),
            "num_steps_random_sparse": int(sparse_summary["num_steps"]),
            "termination_reason_random_sparse": str(sparse_summary["termination_reason"]),
            "wall_clock_sec_random_sparse": float(sparse_summary["wall_clock_sec"]),
        }
    )
    return row


def run_random_sparse_ucr_vs_degree1(args: argparse.Namespace) -> dict[str, Any]:
    reference_csv_path = Path(args.reference_results_csv).expanduser().resolve()
    if not reference_csv_path.exists():
        raise FileNotFoundError(f"Reference results CSV does not exist: {reference_csv_path}")
    reference_summary = load_reference_summary(reference_csv_path, args.reference_summary_json)
    reference_rows = select_reference_rows(
        _load_rows_from_csvs([str(reference_csv_path)]),
        m_values=args.m_values,
        instance_ids=args.instance_ids,
    )
    if not reference_rows:
        raise ValueError("No reference rows selected for random sparse UCR experiment.")

    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir).expanduser().resolve()

    for reference_row in reference_rows:
        n_sys = int(reference_row["n_sys"])
        m_outcome = int(reference_row["M"])
        instance_id = int(reference_row["instance_id"])
        benchmark_seed = int(reference_row["benchmark_seed"])
        data_seed = int(reference_row["data_seed"])
        run_args = build_training_args_from_reference(
            row=reference_row,
            reference_summary=reference_summary,
            cli_args=args,
        )

        problem_args = _problem_namespace(
            n_sys=n_sys,
            m_outcome=m_outcome,
            benchmark_seed=benchmark_seed,
            data_seed=data_seed,
            optimizer="adam",
            learning_rate=float(run_args.learning_rate),
            steps=int(run_args.steps),
            eval_interval=int(run_args.eval_interval),
            threshold=float(run_args.threshold),
            tol=float(run_args.tol),
            su_depth=int(run_args.su_depth),
            scale_init=float(run_args.scale_init),
            bias_scale_init=float(run_args.bias_scale_init),
            weight_decay=0.0,
            state_dtype=str(run_args.state_dtype),
        )
        problem = _build_problem_instance(problem_args)
        target_states = jnp.arange(int(m_outcome), dtype=jnp.int32)
        checkpoint_path = build_restart_checkpoint_path(
            output_dir,
            n_sys=int(n_sys),
            M=int(m_outcome),
            instance_id=int(instance_id),
            model_type=RANDOM_SPARSE_SPEC.model_type,
        )
        sparse_summary, model_restart_rows, mask_payload = _run_random_sparse_restarts(
            problem=problem,
            args=run_args,
            target_states=target_states,
            instance_id=int(instance_id),
            benchmark_seed=int(benchmark_seed),
            data_seed=int(data_seed),
            sparse_seed_offset=int(args.sparse_seed_offset),
            checkpoint_path=checkpoint_path,
        )
        row = build_result_row(
            reference_row=reference_row,
            sparse_summary=sparse_summary,
            mask_payload=mask_payload,
            run_args=run_args,
        )
        rows.append(row)

        for restart_row in model_restart_rows:
            enriched_restart_row = dict(restart_row)
            enriched_restart_row.update(
                {
                    "instance_id": int(instance_id),
                    "n_sys": int(n_sys),
                    "d": int(reference_row["d"]),
                    "M": int(m_outcome),
                    "M_over_d": float(reference_row["M_over_d"]),
                    "benchmark_seed": int(benchmark_seed),
                    "data_seed": int(data_seed),
                    "mask_seed": int(mask_payload["mask_seed"]),
                }
            )
            restart_rows.append(enriched_restart_row)

        mask_record = dict(mask_payload)
        mask_record.update(
            {
                "M_over_d": float(reference_row["M_over_d"]),
                "benchmark_seed": int(benchmark_seed),
                "data_seed": int(data_seed),
            }
        )
        mask_rows.append(mask_record)

        print(
            f"[instance] n_sys={n_sys} d={reference_row['d']} M={m_outcome} instance_id={instance_id} "
            f"gap_full_ref={row['gap_abs_full_ref']:.6f} gap_sparse={row['gap_abs_random_sparse']:.6f}",
            flush=True,
        )

    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        mask_rows=mask_rows,
        args=args,
        output_dir=output_dir,
        reference_summary=reference_summary,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train random sparse UCR baselines using the same instances and training settings "
            "as an existing degree-1 sec5 result CSV."
        )
    )
    parser.add_argument(
        "--reference-results-csv",
        type=str,
        default=str(DEFAULT_REFERENCE_RESULTS_CSV),
    )
    parser.add_argument("--reference-summary-json", type=str, default=None)
    parser.add_argument("--m-values", type=int, nargs="+", default=None)
    parser.add_argument("--instance-ids", type=int, nargs="+", default=None)
    parser.add_argument("--sparse-seed-offset", type=int, default=DEFAULT_SPARSE_SEED_OFFSET)
    parser.add_argument(
        "--random-sparse-execution",
        type=str,
        choices=[RANDOM_SPARSE_EXECUTION_ACTIVE_GATE, RANDOM_SPARSE_EXECUTION_MASKED_ZERO],
        default=DEFAULT_RANDOM_SPARSE_EXECUTION,
    )
    parser.add_argument("--device-name", type=str, default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--diff-method", type=str, default=DEFAULT_DIFF_METHOD)
    parser.add_argument("--jit-backend", type=str, default=DEFAULT_JIT_BACKEND)
    parser.add_argument("--state-dtype", type=str, choices=["complex64", "complex128"], default=DEFAULT_STATE_DTYPE)
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--input-result-csvs", type=str, nargs="+", default=None)
    parser.add_argument("--input-restart-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--input-mask-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    args.random_sparse_execution = normalize_random_sparse_execution(args.random_sparse_execution)
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.aggregate_only:
        return aggregate_existing_outputs(args)
    return run_random_sparse_ucr_vs_degree1(args)


if __name__ == "__main__":
    main()
