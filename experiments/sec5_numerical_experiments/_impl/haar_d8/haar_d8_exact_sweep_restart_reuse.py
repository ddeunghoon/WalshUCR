from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
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
RANDOM_SPARSE_DIR = SEC5_DIR / "random_sparse_model"
SRC_DIR = (SEC5_DIR / "../../../src").resolve()
for path in (CURRENT_DIR, SEC5_DIR, UCR_METHOD_DIR, RANDOM_SPARSE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from walsh_ucr.benchmarks import build_nested_haar_problem
from walsh_ucr.models.vqsd import RandomSparseFullUcrVQSD
from walsh_ucr.training.trainer import TrainResult

from random_sparse_ucr_vs_degree1 import RANDOM_SPARSE_SPEC, build_random_sparse_ucr_mask
from wh_d8_sweep import (
    FULL_UCR_SPEC,
    ModelSpec,
    _append_restart_checkpoint_record,
    _build_model,
    _extract_best_objective,
    _load_restart_checkpoint_records,
    _make_restart_checkpoint_record,
    _resume_state_from_restart_checkpoint_records,
    _stats_payload,
    _termination_reason,
    _validate_restart_checkpoint_records,
    build_projection_groups,
    build_restart_checkpoint_path,
    compute_ucr_parameter_counts,
    make_projected_losses,
    normalize_projection_strategy,
)
from restart_reuse import _make_shared_trainer
from _runner_common import (
    _compute_optimum_success_probability,
    _enrich_restart_rows,
    _ensemble_diagnostics,
    _load_csv_rows,
    _load_jsonl_rows,
    _make_batched_qnode_for_problem,
    _model_result_fields,
    _parse_bool_arg,
    _run_model_restarts,
    _walsh_degree1_parameter_count,
    _write_jsonl,
)


DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "results" / "exact_haar_d8_i10_r50"
DEFAULT_N_SYS_LIST = (3,)
DEFAULT_M_VALUES = (5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_INSTANCE_IDS = tuple(range(10))
DEFAULT_NUM_RESTARTS = 50
DEFAULT_SEED_START = 0
DEFAULT_STEPS = 1000
DEFAULT_EVAL_INTERVAL = 50
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_THRESHOLD = 1e-6
DEFAULT_TOL = 5e-4
DEFAULT_SU_DEPTH = 14
DEFAULT_SCALE_INIT = 1.0
DEFAULT_BIAS_SCALE_INIT = 1.0
DEFAULT_PLOT_DPI = 180
DEFAULT_DISPLAY_FLOOR = 1e-7
DEFAULT_NESTED_MAX_M = 12
DEFAULT_STATE_DTYPE = "complex64"
DEFAULT_ENSEMBLE_MASTER_SEED = 20260504
DEFAULT_SPARSE_SEED_OFFSET = 0
RANDOM_SPARSE_EXECUTION = "active_gate"
ACTIVE_SPARSE_FROZEN_FILL = "not_applicable_active_gate"
BENCHMARK_TYPE = "exact_haar_d8"
ENSEMBLE_NAME = "Exact-Haar-D8"
STATE_GENERATION = "normalized_complex_gaussian"
BASIS_ORDER = "PennyLane/StatePrep computational basis order"
STATEPREP_OPERATION = "qml.StatePrep"
STATEPREP_DECOMPOSITION = "MottonenStatePreparation"
RESULTS_FILENAME = "exact_haar_d8_sweep_results.csv"
RESTARTS_FILENAME = "exact_haar_d8_restart_records.jsonl"
MASKS_FILENAME = "exact_haar_d8_mask_records.jsonl"
SUMMARY_FILENAME = "exact_haar_d8_sweep_summary.json"
GAP_PLOT_FILENAME = "exact_haar_d8_gap_plot.png"
TWO_PANEL_FILENAME = "exact_haar_d8_two_panel.png"

WALSH_DEGREE1_SPEC = ModelSpec(
    model_type="walsh_degree_1",
    model_name="walsh_k_local",
    mean_init="0.0",
    bias_mean_init="pi/2",
    ucr_degree=1,
)
MODEL_SPECS = (FULL_UCR_SPEC, RANDOM_SPARSE_SPEC, WALSH_DEGREE1_SPEC)
GAP_METRICS = (
    "gap_abs_full",
    "gap_abs_random_sparse",
    "gap_abs_walsh_deg1",
)
DIAGNOSTIC_METRICS = (
    "pairwise_fidelity_mean",
    "pairwise_fidelity_std",
    "pairwise_fidelity_max",
    "frame_potential_2",
    "single_qubit_purity_mean",
    "single_qubit_purity_std",
    "gram_rank_numeric",
    "gram_lambda_min",
    "gram_lambda_max",
    "gram_trace",
)


def _resolve_m_values(args: argparse.Namespace) -> list[int]:
    values = [int(value) for value in args.m_values]
    if not values:
        raise ValueError("--m-values must contain at least one value.")
    if min(values) < 2:
        raise ValueError("M values must be >= 2.")
    if max(values) > int(args.nested_max_m):
        raise ValueError(f"M values must be <= nested_max_m={args.nested_max_m}.")
    return values


def _resolve_instance_ids(args: argparse.Namespace) -> list[int]:
    values = [int(value) for value in args.instance_ids]
    if not values:
        raise ValueError("--instance-ids must contain at least one value.")
    if min(values) < 0:
        raise ValueError("instance ids must be >= 0.")
    return values


def _validate_args(args: argparse.Namespace) -> None:
    args.projection_strategy = normalize_projection_strategy(str(args.projection_strategy))
    for n_sys in args.n_sys_list:
        if int(n_sys) != 3:
            raise ValueError(f"Exact Haar D8 benchmark requires n_sys=3, got {n_sys}.")
    if str(args.state_dtype) not in {"complex64", "complex128"}:
        raise ValueError("--state-dtype must be complex64 or complex128.")
    if int(args.num_restarts) < 1:
        raise ValueError("--num-restarts must be >= 1.")
    if int(args.steps) < 1:
        raise ValueError("--steps must be >= 1.")
    if int(args.eval_interval) < 1:
        raise ValueError("--eval-interval must be >= 1.")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if float(args.threshold) < 0.0:
        raise ValueError("--threshold must be >= 0.")
    if float(args.tol) < 0.0:
        raise ValueError("--tol must be >= 0.")
    if float(args.scale_init) < 0.0:
        raise ValueError("--scale-init must be >= 0.")
    if float(args.bias_scale_init) < 0.0:
        raise ValueError("--bias-scale-init must be >= 0.")
    if str(args.optimizer).lower() != "adam":
        raise ValueError("Exact Haar D8 runner supports optimizer='adam' only.")
    if str(args.trainer) != "full":
        raise ValueError("Exact Haar D8 runner supports trainer='full' only.")
    if bool(args.renormalize_projected_probs):
        raise ValueError("Exact Haar plan requires renormalize_projected_probs=False.")
    if args.aggregate_only and not args.input_result_csvs:
        raise ValueError("--aggregate-only requires --input-result-csvs.")
    _resolve_m_values(args)
    _resolve_instance_ids(args)


def _build_problem_instance(
    *,
    n_sys: int,
    M: int,
    instance_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return build_nested_haar_problem(
        benchmark_type=BENCHMARK_TYPE,
        n_sys=int(n_sys),
        M=int(M),
        instance_id=int(instance_id),
        master_seed=int(args.ensemble_master_seed),
        nested_max_m=int(args.nested_max_m),
        state_dtype=str(args.state_dtype),
        fix_global_phase=bool(args.fix_global_phase),
    )


def _gram_diagnostics(states_np: np.ndarray) -> dict[str, Any]:
    states = np.asarray(states_np, dtype=np.complex128)
    gram = states @ np.conj(states.T)
    evals = np.linalg.eigvalsh(gram)
    return {
        "gram_rank_numeric": int(np.linalg.matrix_rank(gram, tol=1e-10)),
        "gram_lambda_min": float(np.min(evals)),
        "gram_lambda_max": float(np.max(evals)),
        "gram_trace": float(np.real(np.trace(gram))),
    }


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
) -> tuple[RandomSparseFullUcrVQSD, dict[str, Any]]:
    full_reference_model = _build_model(
        RANDOM_SPARSE_SPEC,
        n_sys=int(n_sys),
        n_anc=int(n_anc),
        su_depth=int(su_depth),
        scale_init=float(scale_init),
        bias_scale_init=1.0,
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
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION,
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


def _active_random_sparse_checkpoint_records(
    checkpoint_path: Path | None,
    *,
    theta_dim: int,
) -> list[dict[str, Any]]:
    if checkpoint_path is None:
        return []
    active_records: list[dict[str, Any]] = []
    for record in _load_restart_checkpoint_records(checkpoint_path):
        if record.get("random_sparse_execution") != RANDOM_SPARSE_EXECUTION:
            continue
        theta = record.get("theta", [])
        if len(theta) != int(theta_dim):
            raise ValueError(
                "Active random sparse checkpoint has incompatible theta length: "
                f"expected {theta_dim}, got {len(theta)}."
            )
        active_records.append(record)
    return active_records


def _run_active_random_sparse_restarts(
    *,
    problem: dict[str, Any],
    args: argparse.Namespace,
    groups: Sequence[Sequence[int]],
    target_states: jax.Array,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    checkpoint_path: Path | None,
    sparse_seed_offset: int,
    log_prefix: str = "exact-haar",
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
    )
    batched_qnode = _make_batched_qnode_for_problem(
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
    theta_template = model.layout.init_params(jax.random.PRNGKey(int(args.seed_start)))
    trainer = _make_shared_trainer(
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        theta_template=theta_template,
        m_outcome=int(args.m_outcome),
        learning_rate=float(args.learning_rate),
        eval_interval=int(args.eval_interval),
    )
    train_args = (problem["inputs"], target_states)
    eval_args = (problem["inputs"], target_states)

    best_record: dict[str, Any] | None = None
    best_theta: jax.Array | None = None
    restart_records: list[dict[str, Any]] = []
    completed_restart_ids: set[int] = set()

    checkpoint_records = [
        record
        for record in _active_random_sparse_checkpoint_records(
            checkpoint_path,
            theta_dim=int(model.layout.theta_dim),
        )
        if int(record["restart_id"]) < int(args.num_restarts)
    ]
    if checkpoint_records:
        _validate_restart_checkpoint_records(
            checkpoint_records,
            spec=RANDOM_SPARSE_SPEC,
            n_sys=n_sys,
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
            row["random_sparse_execution"] = RANDOM_SPARSE_EXECUTION
        print(
            f"[resume][{log_prefix}][active-random-sparse] n_sys={args.n_sys} "
            f"M={args.m_outcome} instance_id={instance_id} "
            f"completed_restarts={len(completed_restart_ids)}",
            flush=True,
        )

    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_restart_ids:
            continue
        seed_opt = int(args.seed_start) + int(restart_id)
        theta_init = model.layout.init_params(jax.random.PRNGKey(seed_opt))
        opt_state0 = trainer.optimizer.init(theta_init)

        start_time = time.perf_counter()
        result_dict = trainer._solve_adam(
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
        final_objective = float(result.last_eval_loss)
        best_objective = _extract_best_objective(result.loss_log)
        record = {
            "model_type": str(RANDOM_SPARSE_SPEC.model_type),
            "model_name": str(RANDOM_SPARSE_SPEC.model_name),
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION,
            "seed_opt": int(seed_opt),
            "restart_id": int(restart_id),
            "num_steps": int(result.steps_run),
            "termination_reason": _termination_reason(result, max_steps=int(args.steps)),
            "best_objective_value": float(best_objective),
            "final_objective_value": float(final_objective),
            "p_succ": float(1.0 - final_objective),
            "wall_clock_sec": float(wall_clock_sec),
        }
        restart_records.append(record)
        if checkpoint_path is not None:
            checkpoint_record = _make_restart_checkpoint_record(
                record=record,
                theta=result.theta,
                spec=RANDOM_SPARSE_SPEC,
                n_sys=n_sys,
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            checkpoint_record.update(
                {
                    "random_sparse_execution": RANDOM_SPARSE_EXECUTION,
                    "sparse_frozen_fill": ACTIVE_SPARSE_FROZEN_FILL,
                    "theta_dim": int(model.layout.theta_dim),
                    "theta_dim_full_reference": int(mask_payload["theta_dim_full_reference"]),
                }
            )
            _append_restart_checkpoint_record(checkpoint_path, checkpoint_record)
            print(
                f"[restart][checkpointed][{log_prefix}][active-random-sparse] "
                f"n_sys={args.n_sys} M={args.m_outcome} instance_id={instance_id} "
                f"restart_id={restart_id}",
                flush=True,
            )

        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = record
            best_theta = result.theta

    if best_record is None or best_theta is None:
        raise RuntimeError("No valid restart result found for active random_sparse_ucr.")

    return (
        {
            "model_type": str(RANDOM_SPARSE_SPEC.model_type),
            "model_name": str(RANDOM_SPARSE_SPEC.model_name),
            "random_sparse_execution": RANDOM_SPARSE_EXECUTION,
            "p_succ": float(best_record["p_succ"]),
            "best_restart": int(best_record["restart_id"]),
            "seed_opt": int(best_record["seed_opt"]),
            "num_steps": int(best_record["num_steps"]),
            "termination_reason": str(best_record["termination_reason"]),
            "best_objective_value": float(best_record["best_objective_value"]),
            "final_objective_value": float(best_record["final_objective_value"]),
            "wall_clock_sec": float(best_record["wall_clock_sec"]),
            "theta": best_theta,
        },
        restart_records,
        mask_payload,
    )


def _results_fieldnames() -> list[str]:
    return [
        "instance_id",
        "n_sys",
        "d",
        "M",
        "M_over_d",
        "benchmark_type",
        "ensemble_name",
        "state_generation",
        "state_dtype",
        "basis_order",
        "stateprep_operation",
        "stateprep_decomposition",
        "ensemble_seed",
        "benchmark_internal_seed",
        "data_seed",
        "nested_max_m",
        "fix_global_phase",
        "duplicate_policy",
        "prior_type",
        "raw_outcomes",
        "effective_m_outcomes",
        "projection_strategy",
        "class_group_sizes",
        "coverage_ratio",
        "p_opt",
        "pairwise_fidelity_mean",
        "pairwise_fidelity_std",
        "pairwise_fidelity_max",
        "pairwise_fidelity_count",
        "frame_potential_2",
        "single_qubit_purity_mean",
        "single_qubit_purity_std",
        "single_qubit_purity_min",
        "single_qubit_purity_max",
        "gram_rank_numeric",
        "gram_lambda_min",
        "gram_lambda_max",
        "gram_trace",
        "duplicate_fidelity_threshold",
        "duplicate_threshold_exceeded",
        "p_succ_full",
        "gap_abs_full",
        "p_succ_random_sparse",
        "gap_abs_random_sparse",
        "p_succ_walsh_deg1",
        "gap_abs_walsh_deg1",
        "num_ucr_params_full",
        "num_ucr_params_degree1_budget",
        "num_ucr_params_random_sparse",
        "num_ucr_params_walsh_deg1",
        "random_sparse_execution",
        "random_sparse_theta_dim_full_reference",
        "random_sparse_trainable_param_count_total",
        "sparse_budget_rule",
        "sparse_frozen_fill",
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
        "best_restart_walsh_deg1",
        "seed_opt_full",
        "seed_opt_random_sparse",
        "seed_opt_walsh_deg1",
        "num_steps_full",
        "num_steps_random_sparse",
        "num_steps_walsh_deg1",
        "termination_reason_full",
        "termination_reason_random_sparse",
        "termination_reason_walsh_deg1",
        "wall_clock_sec_full",
        "wall_clock_sec_random_sparse",
        "wall_clock_sec_walsh_deg1",
    ]


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = _results_fieldnames()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["M"]), int(item["instance_id"]))):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["M"]), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for M in sorted(grouped):
        bucket = grouped[M]
        first = bucket[0]
        payload: dict[str, Any] = {
            "n_sys": int(first["n_sys"]),
            "d": int(first["d"]),
            "M": int(M),
            "M_over_d": float(first["M_over_d"]),
            "count": int(len(bucket)),
        }
        for metric in ("p_opt", *GAP_METRICS, *DIAGNOSTIC_METRICS):
            stats = _stats_payload([float(row[metric]) for row in bucket])
            for stat_name, stat_value in stats.items():
                payload[f"{metric}_{stat_name}"] = stat_value
        aggregated.append(payload)
    return aggregated


def _fraction_ticklabels(aggregated_rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[str]]:
    tick_pairs = sorted(
        {
            (
                float(row["M_over_d"]),
                (
                    f"{Fraction(int(row['M']), int(row['d'])).numerator}"
                    if Fraction(int(row["M"]), int(row["d"])).denominator == 1
                    else (
                        f"{Fraction(int(row['M']), int(row['d'])).numerator}"
                        f"/{Fraction(int(row['M']), int(row['d'])).denominator}"
                    )
                ),
            )
            for row in aggregated_rows
        },
        key=lambda item: item[0],
    )
    return [item[0] for item in tick_pairs], [item[1] for item in tick_pairs]


def _plot_gap(aggregated_rows: Sequence[dict[str, Any]], path: Path, *, dpi: int) -> None:
    styles = {
        "gap_abs_full": {"label": "full-UCR", "color": "tab:blue", "marker": "o"},
        "gap_abs_random_sparse": {"label": "full-UCR random sparse", "color": "tab:orange", "marker": "v"},
        "gap_abs_walsh_deg1": {"label": "Walsh degree-1", "color": "tab:purple", "marker": "D"},
    }
    xticks, xticklabels = _fraction_ticklabels(aggregated_rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    rows_sorted = sorted(aggregated_rows, key=lambda row: float(row["M_over_d"]))
    x_values = [float(row["M_over_d"]) for row in rows_sorted]
    for metric, style in styles.items():
        ax.errorbar(
            x_values,
            [float(row[f"{metric}_mean"]) for row in rows_sorted],
            yerr=[float(row[f"{metric}_se"]) for row in rows_sorted],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.35,
            markersize=4.0,
            capsize=2.5,
            label=style["label"],
        )
    ax.set_xlabel("M/d")
    ax.set_ylabel("Absolute optimum gap")
    ax.set_xticks(xticks, xticklabels)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_two_panel(aggregated_rows: Sequence[dict[str, Any]], path: Path, *, dpi: int) -> None:
    xticks, xticklabels = _fraction_ticklabels(aggregated_rows)
    rows_sorted = sorted(aggregated_rows, key=lambda row: float(row["M_over_d"]))
    x_values = [float(row["M_over_d"]) for row in rows_sorted]
    fig, (ax_gap, ax_diag) = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    for metric, label, color, marker in (
        ("gap_abs_full", "full-UCR", "tab:blue", "o"),
        ("gap_abs_random_sparse", "full-UCR random sparse", "tab:orange", "v"),
        ("gap_abs_walsh_deg1", "Walsh degree-1", "tab:purple", "D"),
    ):
        ax_gap.errorbar(
            x_values,
            [float(row[f"{metric}_mean"]) for row in rows_sorted],
            yerr=[float(row[f"{metric}_se"]) for row in rows_sorted],
            color=color,
            marker=marker,
            linewidth=1.25,
            markersize=3.5,
            capsize=2.0,
            label=label,
        )
    ax_gap.set_xlabel("M/d")
    ax_gap.set_ylabel("Absolute optimum gap")
    ax_gap.set_xticks(xticks, xticklabels)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(frameon=False, fontsize=7)

    ax_diag.errorbar(
        x_values,
        [float(row["pairwise_fidelity_mean_mean"]) for row in rows_sorted],
        yerr=[float(row["pairwise_fidelity_mean_se"]) for row in rows_sorted],
        color="tab:red",
        marker="s",
        linewidth=1.25,
        markersize=3.5,
        capsize=2.0,
        label="pairwise fidelity mean",
    )
    ax_diag.axhline(1.0 / 8.0, color="0.3", linestyle=":", linewidth=0.9, label="Haar mean 1/d")
    ax_diag.set_xlabel("M/d")
    ax_diag.set_ylabel("Pairwise fidelity")
    ax_diag.set_xticks(xticks, xticklabels)
    ax_diag.grid(True, alpha=0.3)
    ax_diag.legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    raw = output_dir / "raw"
    figures = output_dir / "figures"
    summaries = output_dir / "summaries"
    for path in (raw, figures, summaries):
        path.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "figures": figures, "summaries": summaries}


def _materialize_outputs(
    *,
    rows: Sequence[dict[str, Any]],
    restart_rows: Sequence[dict[str, Any]],
    mask_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    dirs = _prepare_output_dirs(output_dir)
    aggregated_rows = _aggregate_rows(rows)
    raw_csv = dirs["raw"] / RESULTS_FILENAME
    restart_jsonl = dirs["raw"] / RESTARTS_FILENAME
    mask_jsonl = dirs["raw"] / MASKS_FILENAME
    gap_plot = dirs["figures"] / GAP_PLOT_FILENAME
    two_panel = dirs["figures"] / TWO_PANEL_FILENAME
    summary_json = dirs["summaries"] / SUMMARY_FILENAME

    _write_results_csv(raw_csv, rows)
    _write_jsonl(restart_jsonl, restart_rows)
    _write_jsonl(mask_jsonl, mask_rows)
    _plot_gap(aggregated_rows, gap_plot, dpi=int(args.plot_dpi))
    _plot_two_panel(aggregated_rows, two_panel, dpi=int(args.plot_dpi))

    summary = {
        "config": {
            "n_sys_list": [int(value) for value in args.n_sys_list],
            "m_values": [int(value) for value in args.m_values],
            "instance_ids": [int(value) for value in args.instance_ids],
            "num_restarts": int(args.num_restarts),
            "seed_start": int(args.seed_start),
            "steps": int(args.steps),
            "eval_interval": int(args.eval_interval),
            "learning_rate": float(args.learning_rate),
            "threshold": float(args.threshold),
            "tol": float(args.tol),
            "su_depth": int(args.su_depth),
            "scale_init": float(args.scale_init),
            "bias_scale_init": float(args.bias_scale_init),
            "projection_strategy": str(args.projection_strategy),
            "renormalize_projected_probs": bool(args.renormalize_projected_probs),
            "benchmark_type": BENCHMARK_TYPE,
            "ensemble_name": ENSEMBLE_NAME,
            "state_generation": STATE_GENERATION,
            "basis_order": BASIS_ORDER,
            "stateprep_operation": STATEPREP_OPERATION,
            "stateprep_decomposition": STATEPREP_DECOMPOSITION,
            "nested_max_m": int(args.nested_max_m),
            "state_dtype": str(args.state_dtype),
            "fix_global_phase": bool(args.fix_global_phase),
            "ensemble_master_seed": int(args.ensemble_master_seed),
            "duplicate_policy": "record_only",
            "sparse_seed_offset": int(args.sparse_seed_offset),
            "model_types": [str(spec.model_type) for spec in MODEL_SPECS],
        },
        "aggregated_by_m": aggregated_rows,
        "artifacts": {
            "results_csv": str(raw_csv),
            "restart_records_jsonl": str(restart_jsonl),
            "mask_records_jsonl": str(mask_jsonl),
            "gap_plot_png": str(gap_plot),
            "two_panel_png": str(two_panel),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {raw_csv}")
    print(f"saved: {restart_jsonl}")
    print(f"saved: {mask_jsonl}")
    print(f"saved: {gap_plot}")
    print(f"saved: {two_panel}")
    print(f"saved: {summary_json}")
    return summary


def run_exact_haar_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    m_values = _resolve_m_values(args)
    instance_ids = _resolve_instance_ids(args)

    for n_sys in (int(value) for value in args.n_sys_list):
        d = 2 ** int(n_sys)
        args.n_sys = int(n_sys)
        for M in m_values:
            for instance_id in instance_ids:
                problem = _build_problem_instance(
                    n_sys=int(n_sys),
                    M=int(M),
                    instance_id=int(instance_id),
                    args=args,
                )
                args.m_outcome = int(M)
                benchmark_seed = int(problem["ensemble_seed"])
                data_seed = int(problem["data_seed"])
                raw_outcomes = 2 ** int(problem["n_anc"])
                groups, mapping_payload = build_projection_groups(
                    raw_outcomes,
                    int(M),
                    strategy=str(args.projection_strategy),
                )
                p_opt = _compute_optimum_success_probability(
                    states_np=problem["states_np"],
                    M=int(M),
                )
                diagnostics = {
                    **_ensemble_diagnostics(problem["states_np"]),
                    **_gram_diagnostics(problem["states_np"]),
                }
                target_states = jnp.asarray(problem["target_states"], dtype=jnp.int32)

                summaries: dict[str, dict[str, Any]] = {}
                mask_payloads: dict[str, dict[str, Any]] = {}
                for spec in MODEL_SPECS:
                    checkpoint_path = build_restart_checkpoint_path(
                        output_dir,
                        n_sys=int(n_sys),
                        M=int(M),
                        instance_id=int(instance_id),
                        model_type=str(spec.model_type),
                    )
                    if str(spec.model_type) == "random_sparse_ucr":
                        summary, model_restart_rows, mask_payload = _run_active_random_sparse_restarts(
                            problem=problem,
                            args=args,
                            groups=groups,
                            target_states=target_states,
                            instance_id=int(instance_id),
                            benchmark_seed=int(benchmark_seed),
                            data_seed=int(data_seed),
                            checkpoint_path=checkpoint_path,
                            sparse_seed_offset=int(args.sparse_seed_offset),
                            log_prefix="exact-haar",
                        )
                    else:
                        summary, model_restart_rows, mask_payload = _run_model_restarts(
                            spec=spec,
                            problem=problem,
                            args=args,
                            groups=groups,
                            target_states=target_states,
                            instance_id=int(instance_id),
                            benchmark_seed=int(benchmark_seed),
                            data_seed=int(data_seed),
                            checkpoint_path=checkpoint_path,
                            sparse_seed_offset=int(args.sparse_seed_offset),
                            log_prefix="exact-haar",
                        )
                    summaries[str(spec.model_type)] = summary
                    restart_rows.extend(
                        _enrich_restart_rows(
                            model_restart_rows,
                            n_sys=int(n_sys),
                            d=int(d),
                            M=int(M),
                            instance_id=int(instance_id),
                            M_over_d=float(M / d),
                            benchmark_seed=int(benchmark_seed),
                            data_seed=int(data_seed),
                            ensemble_seed=int(problem["ensemble_seed"]),
                        )
                    )
                    if mask_payload is not None:
                        enriched_mask = dict(mask_payload)
                        enriched_mask.update(
                            {
                                "benchmark_type": BENCHMARK_TYPE,
                                "n_sys": int(n_sys),
                                "d": int(d),
                                "M": int(M),
                                "M_over_d": float(M / d),
                                "instance_id": int(instance_id),
                                "benchmark_seed": int(benchmark_seed),
                                "data_seed": int(data_seed),
                                "ensemble_seed": int(problem["ensemble_seed"]),
                            }
                        )
                        mask_payloads[str(spec.model_type)] = enriched_mask
                        mask_rows.append(enriched_mask)

                full_summary = summaries["full_ucr"]
                random_sparse_summary = summaries["random_sparse_ucr"]
                walsh_deg1_summary = summaries["walsh_degree_1"]
                num_full, num_deg1_budget, _ = compute_ucr_parameter_counts(
                    n_sys=int(n_sys),
                    n_anc=int(problem["n_anc"]),
                )
                random_mask = mask_payloads.get("random_sparse_ucr", {})
                row: dict[str, Any] = {
                    "instance_id": int(instance_id),
                    "n_sys": int(n_sys),
                    "d": int(d),
                    "M": int(M),
                    "M_over_d": float(M / d),
                    "benchmark_type": BENCHMARK_TYPE,
                    "ensemble_name": ENSEMBLE_NAME,
                    "state_generation": STATE_GENERATION,
                    "state_dtype": str(args.state_dtype),
                    "basis_order": BASIS_ORDER,
                    "stateprep_operation": STATEPREP_OPERATION,
                    "stateprep_decomposition": STATEPREP_DECOMPOSITION,
                    "ensemble_seed": int(problem["ensemble_seed"]),
                    "benchmark_internal_seed": int(problem["benchmark_internal_seed"]),
                    "data_seed": int(problem["data_seed"]),
                    "nested_max_m": int(args.nested_max_m),
                    "fix_global_phase": bool(args.fix_global_phase),
                    "duplicate_policy": "record_only",
                    "prior_type": "uniform",
                    "raw_outcomes": int(raw_outcomes),
                    "effective_m_outcomes": int(M),
                    "projection_strategy": str(mapping_payload["strategy"]),
                    "class_group_sizes": json.dumps(mapping_payload["class_group_sizes"]),
                    "coverage_ratio": float(mapping_payload["coverage_ratio"]),
                    "p_opt": float(p_opt),
                    **diagnostics,
                    **_model_result_fields("full", full_summary),
                    **_model_result_fields("random_sparse", random_sparse_summary),
                    **_model_result_fields("walsh_deg1", walsh_deg1_summary),
                    "gap_abs_full": float(p_opt - float(full_summary["p_succ"])),
                    "gap_abs_random_sparse": float(p_opt - float(random_sparse_summary["p_succ"])),
                    "gap_abs_walsh_deg1": float(p_opt - float(walsh_deg1_summary["p_succ"])),
                    "num_ucr_params_full": int(num_full),
                    "num_ucr_params_degree1_budget": int(num_deg1_budget),
                    "num_ucr_params_random_sparse": int(random_mask.get("num_ucr_params_sparse", 0)),
                    "num_ucr_params_walsh_deg1": _walsh_degree1_parameter_count(
                        n_sys=int(n_sys),
                        n_anc=int(problem["n_anc"]),
                    ),
                    "random_sparse_execution": str(
                        random_mask.get("random_sparse_execution", RANDOM_SPARSE_EXECUTION)
                    ),
                    "random_sparse_theta_dim_full_reference": int(
                        random_mask.get("theta_dim_full_reference", 0)
                    ),
                    "random_sparse_trainable_param_count_total": int(
                        random_mask.get("trainable_param_count_total", 0)
                    ),
                    "sparse_budget_rule": str(random_mask.get("sparse_budget_rule", "blockwise_match_wd1_param_count")),
                    "sparse_frozen_fill": str(
                        random_mask.get("sparse_frozen_fill", ACTIVE_SPARSE_FROZEN_FILL)
                    ),
                    "optimizer_name": "adam",
                    "learning_rate": float(args.learning_rate),
                    "learning_rate_schedule": "constant",
                    "max_steps": int(args.steps),
                    "eval_interval": int(args.eval_interval),
                    "threshold": float(args.threshold),
                    "num_restarts": int(args.num_restarts),
                    "stopping_rule": "abs_eval_loss_delta_le_threshold_on_eval_interval",
                    "aggregation_rule": "best_over_restarts_for_row;mean_se_for_figure;median_iqr_in_summary",
                }
                rows.append(row)
                print(
                    f"[exact-haar-instance] n_sys={n_sys} M={M} instance_id={instance_id} "
                    f"gap_full={row['gap_abs_full']:.6f} "
                    f"gap_random_sparse={row['gap_abs_random_sparse']:.6f} "
                    f"gap_walsh_deg1={row['gap_abs_walsh_deg1']:.6f}",
                    flush=True,
                )

    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        mask_rows=mask_rows,
        args=args,
        output_dir=output_dir,
    )


def aggregate_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    rows = _load_csv_rows(args.input_result_csvs)
    restart_rows = _load_jsonl_rows(args.input_restart_jsonls)
    mask_rows = _load_jsonl_rows(args.input_mask_jsonls)
    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        mask_rows=mask_rows,
        args=args,
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the d=8 exact Haar pure-state Section 5 benchmark with restart reuse."
    )
    parser.add_argument("--n-sys-list", type=int, nargs="+", default=list(DEFAULT_N_SYS_LIST))
    parser.add_argument("--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    parser.add_argument("--instance-ids", type=int, nargs="+", default=list(DEFAULT_INSTANCE_IDS))
    parser.add_argument("--num-restarts", type=int, default=DEFAULT_NUM_RESTARTS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--trainer", type=str, choices=["full"], default="full")
    parser.add_argument("--loss-type", type=str, choices=["linear", "js", "nll"], default="linear")
    parser.add_argument("--device-name", type=str, default="default.qubit")
    parser.add_argument("--diff-method", type=str, default="backprop")
    parser.add_argument("--jit-backend", type=str, default="gpu")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--su-depth", type=int, default=DEFAULT_SU_DEPTH)
    parser.add_argument("--scale-init", type=float, default=DEFAULT_SCALE_INIT)
    parser.add_argument("--bias-scale-init", type=float, default=DEFAULT_BIAS_SCALE_INIT)
    parser.add_argument("--projection-strategy", type=str, default="drop_extra")
    parser.add_argument(
        "--renormalize-projected-probs",
        type=_parse_bool_arg,
        default=False,
        metavar="{True,False}",
    )
    parser.add_argument("--state-dtype", type=str, choices=["complex64", "complex128"], default=DEFAULT_STATE_DTYPE)
    parser.add_argument("--nested-max-m", type=int, default=DEFAULT_NESTED_MAX_M)
    parser.add_argument("--fix-global-phase", type=_parse_bool_arg, default=False, metavar="{True,False}")
    parser.add_argument("--ensemble-master-seed", type=int, default=DEFAULT_ENSEMBLE_MASTER_SEED)
    parser.add_argument("--sparse-seed-offset", type=int, default=DEFAULT_SPARSE_SEED_OFFSET)
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--display-floor", type=float, default=DEFAULT_DISPLAY_FLOOR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--input-result-csvs", type=str, nargs="+", default=None)
    parser.add_argument("--input-restart-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--input-mask-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.aggregate_only):
        aggregate_existing_outputs(args)
        return
    run_exact_haar_sweep(args)


if __name__ == "__main__":
    main()
