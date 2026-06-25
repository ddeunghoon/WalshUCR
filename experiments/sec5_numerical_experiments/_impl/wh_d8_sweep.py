from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

CURRENT_DIR = Path(__file__).resolve().parent
UCR_METHOD_DIR = CURRENT_DIR.parent
SRC_DIR = (CURRENT_DIR / "../../../src").resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(UCR_METHOD_DIR) not in sys.path:
    sys.path.append(str(UCR_METHOD_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from walsh_ucr.engine import make_batched_qnode
from walsh_ucr.models.vqsd import FullUcrVQSD, WalshKLocalVQSD
from walsh_ucr.training.trainer import JAX_Debug_Trainer, JAX_Full_Trainer

from common import (
    _fieldnames_with_extras,
    _load_jsonl_rows,
    _prepare_output_dirs,
    _read_typed_csv_rows,
    _write_dict_csv,
    _write_jsonl,
)
from weyl_problem import _build_problem_instance, _compute_sdp_value, _parse_bool_arg
from weyl_statevector_backend import make_weyl_statevector_batched_qnode, normalize_simulation_backend


DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "results" / "wh_md_sweep"
DEFAULT_N_SYS_LIST = (3, 4)
DEFAULT_NUM_INSTANCES = 10
DEFAULT_NUM_RESTARTS = 30
DEFAULT_STEPS = 1000
DEFAULT_EVAL_INTERVAL = 50
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_THRESHOLD = 1e-6
DEFAULT_TOL = 5e-4
DEFAULT_SU_DEPTH = 14
DEFAULT_SCALE_INIT = 0.01
DEFAULT_BIAS_SCALE_INIT = 0.01
DEFAULT_PLOT_DPI = 200
DEFAULT_SEED_START = 0
RESTART_CHECKPOINT_VERSION = 1
PROJECTION_STRATEGY_ALIASES = {
    "balanced_tail_block": "balanced_tail_block",
    "partition_all": "balanced_tail_block",
    "drop_extra": "drop_extra",
}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_name: str
    mean_init: str
    bias_mean_init: str | None
    ucr_degree: int | None = None


FULL_UCR_SPEC = ModelSpec(
    model_type="full_ucr",
    model_name="vqsd",
    mean_init="pi/2",
    bias_mean_init=None,
)
MODEL_SPECS = (FULL_UCR_SPEC,)


def build_boundary_centered_m_grid(n_sys_list: Sequence[int]) -> dict[int, list[int]]:
    grid: dict[int, list[int]] = {}
    for n_sys in n_sys_list:
        d = 2 ** int(n_sys)
        grid[d] = [d - 2, d - 1, d, d + 1, d + 2]
    return grid


def resolve_m_grid(args: argparse.Namespace) -> dict[int, list[int]]:
    if args.m_values:
        m_values = [int(value) for value in args.m_values]
        if any(value < 2 for value in m_values):
            raise ValueError(f"All m_values must be >= 2, got {m_values}.")
        return {2 ** int(n_sys): list(m_values) for n_sys in args.n_sys_list}
    return build_boundary_centered_m_grid(args.n_sys_list)


def resolve_instance_ids(args: argparse.Namespace) -> list[int]:
    if args.instance_ids:
        instance_ids = sorted({int(value) for value in args.instance_ids})
        if any(value < 0 for value in instance_ids):
            raise ValueError(f"instance_ids must be >= 0, got {instance_ids}.")
        return instance_ids
    return list(range(int(args.num_instances_per_grid_point)))


def normalize_projection_strategy(value: str) -> str:
    normalized = str(value).strip().lower()
    try:
        return PROJECTION_STRATEGY_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PROJECTION_STRATEGY_ALIASES))
        raise ValueError(
            f"Unsupported projection_strategy '{value}'. Supported values: {supported}."
        ) from exc


def projection_strategy_projects_all_raw_outcomes(strategy: str) -> bool:
    return normalize_projection_strategy(strategy) == "balanced_tail_block"


def build_balanced_tail_block_groups(raw_outcomes: int, m_outcomes: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    raw = int(raw_outcomes)
    m = int(m_outcomes)
    if raw < 1:
        raise ValueError(f"raw_outcomes must be >= 1, got {raw_outcomes}.")
    if m < 1:
        raise ValueError(f"m_outcomes must be >= 1, got {m_outcomes}.")
    if raw < m:
        raise ValueError(
            f"balanced_tail_block requires raw_outcomes >= m_outcomes, got raw={raw}, m={m}."
        )

    base_size, remainder = divmod(raw, m)
    sizes = [base_size] * m
    for idx in range(remainder):
        sizes[m - remainder + idx] += 1

    groups: list[np.ndarray] = []
    start = 0
    for size in sizes:
        stop = start + size
        groups.append(np.arange(start, stop, dtype=np.int32))
        start = stop

    assigned = set()
    for group in groups:
        assigned.update(int(value) for value in group.tolist())

    payload = {
        "strategy": "balanced_tail_block",
        "raw_outcomes": raw,
        "m_outcomes": m,
        "class_to_outcomes": [group.tolist() for group in groups],
        "class_group_sizes": [int(group.shape[0]) for group in groups],
        "assigned_raw_outcomes": sorted(assigned),
        "unassigned_raw_outcomes": sorted(set(range(raw)) - assigned),
        "coverage_ratio": float(len(assigned) / float(raw)),
    }
    return groups, payload


def build_drop_extra_groups(raw_outcomes: int, m_outcomes: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    raw = int(raw_outcomes)
    m = int(m_outcomes)
    if raw < 1:
        raise ValueError(f"raw_outcomes must be >= 1, got {raw_outcomes}.")
    if m < 1:
        raise ValueError(f"m_outcomes must be >= 1, got {m_outcomes}.")
    if raw < m:
        raise ValueError(f"drop_extra requires raw_outcomes >= m_outcomes, got raw={raw}, m={m}.")

    groups = [np.asarray([idx], dtype=np.int32) for idx in range(m)]
    assigned = list(range(m))
    payload = {
        "strategy": "drop_extra",
        "raw_outcomes": raw,
        "m_outcomes": m,
        "class_to_outcomes": [group.tolist() for group in groups],
        "class_group_sizes": [1] * m,
        "assigned_raw_outcomes": assigned,
        "unassigned_raw_outcomes": list(range(m, raw)),
        "coverage_ratio": float(m / float(raw)),
    }
    return groups, payload


def build_projection_groups(
    raw_outcomes: int,
    m_outcomes: int,
    strategy: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    resolved = normalize_projection_strategy(strategy)
    if resolved == "balanced_tail_block":
        return build_balanced_tail_block_groups(raw_outcomes, m_outcomes)
    if resolved == "drop_extra":
        return build_drop_extra_groups(raw_outcomes, m_outcomes)
    raise ValueError(f"Unsupported normalized projection strategy '{resolved}'.")


def make_projector(groups: Sequence[np.ndarray]) -> Callable[[jax.Array], jax.Array]:
    group_idx = tuple(jnp.asarray(group, dtype=jnp.int32) for group in groups)

    def projector(raw_probs: jax.Array) -> jax.Array:
        cols = []
        for idx in group_idx:
            cols.append(jnp.sum(raw_probs[:, idx], axis=1))
        return jnp.stack(cols, axis=1)

    return projector


def make_projected_losses(
    batched_qnode: Callable[[Any, jax.Array], jax.Array],
    *,
    groups: Sequence[np.ndarray],
    loss_type: str = "linear",
    renormalize_projected_probs: bool = False,
    eps: float = 1e-12,
) -> tuple[Callable[..., jax.Array], Callable[..., tuple[jax.Array, jax.Array, jax.Array]], Callable[[jax.Array], jax.Array]]:
    projector = make_projector(groups)
    eps_arr = jnp.asarray(eps, dtype=jnp.float64)

    def _effective_probs(params: jax.Array, inputs: Any) -> tuple[jax.Array, jax.Array]:
        raw_probs = batched_qnode(inputs, params)
        class_probs = projector(raw_probs)
        if renormalize_projected_probs:
            assigned_mass = jnp.sum(class_probs, axis=1, keepdims=True)
            class_probs = class_probs / jnp.clip(assigned_mass, a_min=eps_arr)
        return class_probs, raw_probs

    def forward_eval(params: jax.Array, inputs: Any, target_states: jax.Array | None = None):
        class_probs, raw_probs = _effective_probs(params, inputs)
        m = class_probs.shape[0]
        if target_states is None:
            target_states = jnp.arange(m, dtype=jnp.int32)
        idx = jnp.arange(m, dtype=jnp.int32)
        p_correct = class_probs[idx, target_states]
        loss = 1.0 - jnp.mean(p_correct)
        return loss, class_probs, raw_probs

    def train_loss_linear(params: jax.Array, inputs: Any, target_states: jax.Array) -> jax.Array:
        loss, _, _ = forward_eval(params, inputs, target_states)
        return loss

    def train_loss_nll(params: jax.Array, inputs: Any, target_states: jax.Array) -> jax.Array:
        class_probs, _ = _effective_probs(params, inputs)
        idx = jnp.arange(class_probs.shape[0], dtype=jnp.int32)
        p_correct = jnp.clip(class_probs[idx, target_states], eps_arr, 1.0)
        return -jnp.mean(jnp.log(p_correct))

    def train_loss_js(params: jax.Array, inputs: Any, target_states: jax.Array) -> jax.Array:
        class_probs, _ = _effective_probs(params, inputs)
        class_probs = jnp.clip(class_probs, eps_arr, 1.0)
        _, c = class_probs.shape
        onehot = jax.nn.one_hot(target_states, c)
        alpha = jnp.asarray(0.01, dtype=class_probs.dtype)
        q = (1.0 - alpha) * onehot + alpha / c
        q = jnp.clip(q, eps_arr, 1.0)
        m = 0.5 * (q + class_probs)
        js = 0.5 * jnp.sum(q * (jnp.log(q) - jnp.log(m)), axis=-1) + 0.5 * jnp.sum(
            class_probs * (jnp.log(class_probs) - jnp.log(m)),
            axis=-1,
        )
        return jnp.mean(js)

    if loss_type == "linear":
        train_loss_fn = train_loss_linear
    elif loss_type == "nll":
        train_loss_fn = train_loss_nll
    elif loss_type == "js":
        train_loss_fn = train_loss_js
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'.")

    return train_loss_fn, forward_eval, projector


def _problem_namespace(
    *,
    n_sys: int,
    m_outcome: int,
    benchmark_seed: int,
    data_seed: int,
    optimizer: str,
    learning_rate: float,
    steps: int,
    eval_interval: int,
    threshold: float,
    tol: float,
    su_depth: int,
    scale_init: float,
    bias_scale_init: float,
    weight_decay: float,
    state_dtype: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        worker=False,
        aggregate=False,
        run_idx=None,
        n_sys=int(n_sys),
        problem_type="weyl",
        m_outcome=int(m_outcome),
        canonical_labels=False,
        k_inits=1,
        seed_start=DEFAULT_SEED_START,
        benchmark_seed=int(benchmark_seed),
        data_seed=int(data_seed),
        state_seed=1,
        state_dtype=str(state_dtype),
        su_depth=int(su_depth),
        model="vqsd",
        mean_init="pi/2",
        bias_mean_init="pi/2",
        scale_init=float(scale_init),
        bias_scale_init=float(bias_scale_init),
        steps=int(steps),
        eval_interval=int(eval_interval),
        optimizer=str(optimizer),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        trainer="full",
        loss_type="linear",
        device_name="default.qubit",
        diff_method="backprop",
        jit_backend="gpu",
        threshold=float(threshold),
        tol=float(tol),
        jobs=1,
        output_dir=None,
    )


def _build_model(
    spec: ModelSpec,
    *,
    n_sys: int,
    n_anc: int,
    su_depth: int,
    scale_init: float,
    bias_scale_init: float,
) -> FullUcrVQSD | WalshKLocalVQSD:
    if spec.model_name == "vqsd":
        return FullUcrVQSD(
            n_anc=int(n_anc),
            n_sys=int(n_sys),
            su_depth=int(su_depth),
            mean_init=str(spec.mean_init),
            scale_init=float(scale_init),
        )
    if spec.model_name == "walsh_k_local":
        if spec.ucr_degree is None:
            raise ValueError("walsh_k_local ModelSpec requires ucr_degree.")
        return WalshKLocalVQSD(
            n_anc=int(n_anc),
            n_sys=int(n_sys),
            su_depth=int(su_depth),
            ucr_degree=int(spec.ucr_degree),
            mean_init=str(spec.mean_init),
            scale_init=float(scale_init),
            bias_mean_init=str(spec.bias_mean_init if spec.bias_mean_init is not None else "pi/2"),
            bias_scale_init=float(bias_scale_init),
        )
    raise ValueError(f"Unsupported sec5 model '{spec.model_name}'.")


def _build_batched_qnode_for_problem(
    *,
    problem: dict[str, Any],
    model: FullUcrVQSD | WalshKLocalVQSD,
    n_sys: int,
    device_name: str,
    diff_method: str,
    simulation_backend: str = "pennylane",
) -> Callable[[Any, jax.Array], jax.Array]:
    if str(problem["problem_type"]) != "weyl":
        raise ValueError("Section 5.1 WH sweep only supports Weyl problems.")

    backend = normalize_simulation_backend(simulation_backend)
    if backend == "jax_statevector":
        if isinstance(model, WalshKLocalVQSD):
            raise ValueError(
                "jax_statevector backend currently supports FullUcrVQSD only. "
                "Use simulation_backend='pennylane' for WalshKLocalVQSD."
            )
        return make_weyl_statevector_batched_qnode(
            problem=problem,
            model=model,
            n_sys=int(n_sys),
        )

    n_anc = int(problem["n_anc"])
    sys_wires = list(range(int(n_sys)))
    anc_wires = list(range(int(n_sys), int(n_sys) + n_anc))
    all_wires = sys_wires + anc_wires
    return make_batched_qnode(
        n_sys=int(n_sys),
        sys_wires=sys_wires,
        anc_wires=anc_wires,
        all_wires=all_wires,
        model=model,
        benchmark=problem["benchmark"],
        device_name=str(device_name),
        diff_method=str(diff_method),
    )


def compute_ucr_parameter_counts(*, n_sys: int, n_anc: int) -> tuple[int, int, float]:
    num_full = sum(2 ** (int(n_sys) + block_idx) for block_idx in range(int(n_anc)))
    num_degree1_budget = sum(1 + int(n_sys) + block_idx for block_idx in range(int(n_anc)))
    return int(num_full), int(num_degree1_budget), float(num_full / num_degree1_budget)


def compute_optimum_success_probability(*, problem: dict[str, Any], n_sys: int, m_outcome: int) -> float:
    sdp_error = float(
        _compute_sdp_value(
            problem=problem,
            n_sys=int(n_sys),
            m_outcome=int(m_outcome),
        )
    )
    return float(1.0 - sdp_error)


def _termination_reason(result: Any, *, max_steps: int) -> str:
    if bool(result.nan_found):
        return "nan"
    if bool(result.stopped_early):
        return "threshold"
    if int(result.steps_run) >= int(max_steps):
        return "max_steps"
    return "completed"


def _extract_best_objective(loss_log: Any) -> float:
    values = np.asarray(loss_log, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


def _training_metadata_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "optimizer_name": "adam",
        "learning_rate": float(args.learning_rate),
        "learning_rate_schedule": "constant",
        "max_steps": int(args.steps),
        "eval_interval": int(args.eval_interval),
        "threshold": float(args.threshold),
        "num_restarts": int(args.num_restarts),
        "stopping_rule": "abs_eval_loss_delta_le_threshold_on_eval_interval",
        "aggregation_rule": "best_over_restarts_for_row;mean_se_for_figure;median_iqr_in_summary",
        "loss_type": str(args.loss_type),
        "prior_type": "uniform",
        "use_scrambler": bool(args.use_scrambler),
    }


def _run_model_restarts(
    *,
    spec: ModelSpec,
    problem: dict[str, Any],
    args: argparse.Namespace,
    groups: Sequence[np.ndarray],
    target_states: jax.Array,
    checkpoint_path: Path | None = None,
    instance_id: int | None = None,
    benchmark_seed: int | None = None,
    data_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_sys = int(args.n_sys)
    n_anc = int(problem["n_anc"])
    model = _build_model(
        spec,
        n_sys=n_sys,
        n_anc=n_anc,
        su_depth=int(args.su_depth),
        scale_init=float(args.scale_init),
        bias_scale_init=float(args.bias_scale_init),
    )
    batched_qnode = _build_batched_qnode_for_problem(
        problem=problem,
        model=model,
        n_sys=n_sys,
        device_name=str(args.device_name),
        diff_method=str(args.diff_method),
        simulation_backend=str(getattr(args, "simulation_backend", "pennylane")),
    )
    train_loss_fn, eval_loss_fn, _ = make_projected_losses(
        batched_qnode,
        groups=groups,
        loss_type=str(args.loss_type),
        renormalize_projected_probs=bool(args.renormalize_projected_probs),
    )
    trainer_cls = JAX_Full_Trainer if str(args.trainer) == "full" else JAX_Debug_Trainer
    a_priori_probs = jnp.ones((int(args.m_outcome),), dtype=jnp.float64) / float(args.m_outcome)

    best_record: dict[str, Any] | None = None
    best_theta: jax.Array | None = None
    restart_records: list[dict[str, Any]] = []
    completed_restart_ids: set[int] = set()

    if checkpoint_path is not None:
        checkpoint_records = [
            record
            for record in _load_restart_checkpoint_records(checkpoint_path)
            if int(record["restart_id"]) < int(args.num_restarts)
        ]
        if checkpoint_records:
            if instance_id is None or benchmark_seed is None or data_seed is None:
                raise ValueError("Checkpoint resume requires instance_id, benchmark_seed, and data_seed.")
            _validate_restart_checkpoint_records(
                checkpoint_records,
                spec=spec,
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
            print(
                f"[resume] model={spec.model_type} n_sys={args.n_sys} M={args.m_outcome} "
                f"instance_id={instance_id} completed_restarts={len(completed_restart_ids)}",
                flush=True,
            )

    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_restart_ids:
            continue
        seed_opt = int(args.seed_start) + restart_id
        theta_init = model.layout.init_params(jax.random.PRNGKey(seed_opt))
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
            "model_type": spec.model_type,
            "model_name": spec.model_name,
            "seed_opt": int(seed_opt),
            "restart_id": int(restart_id),
            "num_steps": int(result.steps_run),
            "termination_reason": _termination_reason(result, max_steps=int(args.steps)),
            "best_objective_value": float(best_objective),
            "final_objective_value": float(final_objective),
            "p_succ": float(p_succ),
            "wall_clock_sec": float(wall_clock_sec),
            "simulation_backend": normalize_simulation_backend(getattr(args, "simulation_backend", "pennylane")),
        }
        restart_records.append(record)
        if checkpoint_path is not None:
            if instance_id is None or benchmark_seed is None or data_seed is None:
                raise ValueError("Checkpoint write requires instance_id, benchmark_seed, and data_seed.")
            checkpoint_record = _make_restart_checkpoint_record(
                record=record,
                theta=result.theta,
                spec=spec,
                n_sys=int(args.n_sys),
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            _append_restart_checkpoint_record(checkpoint_path, checkpoint_record)
            print(
                f"[restart][checkpointed] model={spec.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} restart_id={restart_id}",
                flush=True,
            )

        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = record
            best_theta = result.theta

    if best_record is None or best_theta is None:
        raise RuntimeError(f"No valid restart result found for model '{spec.model_type}'.")

    summary = {
        "model_type": spec.model_type,
        "model_name": spec.model_name,
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
    return summary, restart_records


def _stats_payload(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "se": float(np.std(arr) / math.sqrt(arr.size)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["d"]), int(row["M"]))
        grouped.setdefault(key, []).append(row)

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
        for column in ("gap_abs_full",):
            stats = _stats_payload([float(row[column]) for row in bucket])
            for stat_name, stat_value in stats.items():
                payload[f"{column}_{stat_name}"] = stat_value
        aggregated_rows.append(payload)

    regime_summary: dict[str, Any] = {}
    for regime_name, predicate in (
        ("M_over_d_le_1", lambda row: float(row["M_over_d"]) <= 1.0),
        ("M_over_d_gt_1", lambda row: float(row["M_over_d"]) > 1.0),
    ):
        bucket = [row for row in rows if predicate(row)]
        regime_payload = {
            "count": int(len(bucket)),
            "gap_abs_full": _stats_payload([float(row["gap_abs_full"]) for row in bucket]),
        }
        regime_summary[regime_name] = regime_payload

    return aggregated_rows, regime_summary


def _plot_gap(aggregated_rows: Sequence[dict[str, Any]], path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    line_styles = {3: "-", 4: "--"}

    for n_sys in sorted({int(row["n_sys"]) for row in aggregated_rows}):
        rows_n = sorted(
            [row for row in aggregated_rows if int(row["n_sys"]) == n_sys],
            key=lambda row: float(row["M_over_d"]),
        )
        x = np.asarray([float(row["M_over_d"]) for row in rows_n], dtype=np.float64)
        d = int(rows_n[0]["d"])
        y = np.asarray([float(row["gap_abs_full_mean"]) for row in rows_n], dtype=np.float64)
        yerr = np.asarray([float(row["gap_abs_full_se"]) for row in rows_n], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color="tab:blue",
            marker="o",
            linestyle=line_styles.get(n_sys, "-."),
            linewidth=1.6,
            markersize=5.5,
            capsize=3.0,
            label=f"full-UCR (d={d})",
        )

    ax.set_xlabel("M/d")
    ax.set_ylabel("Absolute optimum gap")
    ax.set_title("Section 5.1: WH optimum gap sweep")
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
        "p_succ_full",
        "gap_abs_full",
        "gap_rel_full",
        "num_ucr_params_full",
        "num_ucr_params_degree1_budget",
        "ucr_compression_ratio",
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
        "num_steps_full",
        "termination_reason_full",
        "wall_clock_sec_full",
        "simulation_backend",
    ]


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_dict_csv(path, rows, fieldnames=_fieldnames_with_extras(rows, _results_fieldnames()))


def _write_restart_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def build_restart_checkpoint_path(
    output_dir: Path,
    *,
    n_sys: int,
    M: int,
    instance_id: int,
    model_type: str,
) -> Path:
    checkpoint_dir = output_dir / "raw" / "restart_checkpoints"
    return checkpoint_dir / (
        f"nsys{int(n_sys)}_M{int(M)}_instance{int(instance_id):02d}_{str(model_type)}.jsonl"
    )


def _restart_checkpoint_scalar_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model_type": str(record["model_type"]),
        "model_name": str(record["model_name"]),
        "seed_opt": int(record["seed_opt"]),
        "restart_id": int(record["restart_id"]),
        "num_steps": int(record["num_steps"]),
        "termination_reason": str(record["termination_reason"]),
        "best_objective_value": float(record["best_objective_value"]),
        "final_objective_value": float(record["final_objective_value"]),
        "p_succ": float(record["p_succ"]),
        "wall_clock_sec": float(record["wall_clock_sec"]),
    }
    if "simulation_backend" in record:
        payload["simulation_backend"] = normalize_simulation_backend(record["simulation_backend"])
    return payload


def _make_restart_checkpoint_record(
    *,
    record: dict[str, Any],
    theta: jax.Array,
    spec: ModelSpec,
    n_sys: int,
    M: int,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _restart_checkpoint_scalar_record(record)
    payload.update(
        {
            "checkpoint_version": int(RESTART_CHECKPOINT_VERSION),
            "n_sys": int(n_sys),
            "M": int(M),
            "instance_id": int(instance_id),
            "benchmark_seed": int(benchmark_seed),
            "data_seed": int(data_seed),
            "projection_strategy": str(args.projection_strategy),
            "learning_rate": float(args.learning_rate),
            "max_steps": int(args.steps),
            "eval_interval": int(args.eval_interval),
            "threshold": float(args.threshold),
            "su_depth": int(args.su_depth),
            "simulation_backend": normalize_simulation_backend(getattr(args, "simulation_backend", "pennylane")),
            "theta": np.asarray(theta, dtype=np.float64).reshape(-1).tolist(),
        }
    )
    payload["model_type"] = str(spec.model_type)
    payload["model_name"] = str(spec.model_name)
    return payload


def _append_restart_checkpoint_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_restart_checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows_by_restart_id: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"warning: ignoring malformed checkpoint line {lineno} in {path}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            rows_by_restart_id[int(payload["restart_id"])] = payload

    return [rows_by_restart_id[key] for key in sorted(rows_by_restart_id)]


def _validate_restart_checkpoint_records(
    records: Sequence[dict[str, Any]],
    *,
    spec: ModelSpec,
    n_sys: int,
    M: int,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    args: argparse.Namespace,
) -> None:
    expected_pairs = {
        "checkpoint_version": int(RESTART_CHECKPOINT_VERSION),
        "model_type": str(spec.model_type),
        "model_name": str(spec.model_name),
        "n_sys": int(n_sys),
        "M": int(M),
        "instance_id": int(instance_id),
        "benchmark_seed": int(benchmark_seed),
        "data_seed": int(data_seed),
        "projection_strategy": str(args.projection_strategy),
        "learning_rate": float(args.learning_rate),
        "max_steps": int(args.steps),
        "eval_interval": int(args.eval_interval),
        "threshold": float(args.threshold),
        "su_depth": int(args.su_depth),
    }
    for record in records:
        if "simulation_backend" in record:
            normalize_simulation_backend(record["simulation_backend"])
        for key, expected_value in expected_pairs.items():
            if record.get(key) != expected_value:
                raise ValueError(
                    f"Checkpoint mismatch for {key}: expected {expected_value}, got {record.get(key)}."
                )


def _resume_state_from_restart_checkpoint_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, jax.Array | None, set[int]]:
    restart_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_theta: jax.Array | None = None
    completed_restart_ids: set[int] = set()

    for checkpoint_record in records:
        scalar_record = _restart_checkpoint_scalar_record(checkpoint_record)
        restart_records.append(scalar_record)
        completed_restart_ids.add(int(scalar_record["restart_id"]))
        final_objective = float(scalar_record["final_objective_value"])
        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = scalar_record
            best_theta = jnp.asarray(checkpoint_record["theta"], dtype=jnp.float64)

    return restart_records, best_record, best_theta, completed_restart_ids


def _coerce_csv_value(key: str, value: str) -> Any:
    if key in {
        "instance_id",
        "n_sys",
        "d",
        "M",
        "benchmark_seed",
        "data_seed",
        "raw_outcomes",
        "effective_m_outcomes",
        "num_ucr_params_full",
        "num_ucr_params_degree1_budget",
        "best_restart_full",
        "num_steps_full",
        "num_restarts",
        "max_steps",
        "eval_interval",
    }:
        return int(value)
    if key in {
        "M_over_d",
        "coverage_ratio",
        "p_opt",
        "p_succ_full",
        "gap_abs_full",
        "gap_rel_full",
        "ucr_compression_ratio",
        "learning_rate",
        "threshold",
        "wall_clock_sec_full",
    }:
        return float(value)
    return value


def _load_rows_from_csvs(paths: Sequence[str]) -> list[dict[str, Any]]:
    return _read_typed_csv_rows(paths, _coerce_csv_value)


def _load_restart_rows_from_jsonls(paths: Sequence[str]) -> list[dict[str, Any]]:
    return _load_jsonl_rows(paths)


def _validate_args(args: argparse.Namespace) -> None:
    args.projection_strategy = normalize_projection_strategy(args.projection_strategy)
    args.simulation_backend = normalize_simulation_backend(getattr(args, "simulation_backend", "pennylane"))
    for value in args.n_sys_list:
        if int(value) < 1:
            raise ValueError(f"n_sys entries must be >= 1, got {value}.")
    if int(args.num_instances_per_grid_point) < 1:
        raise ValueError("num_instances_per_grid_point must be >= 1.")
    if int(args.num_restarts) < 1:
        raise ValueError("num_restarts must be >= 1.")
    if int(args.steps) < 1:
        raise ValueError("steps must be >= 1.")
    if int(args.eval_interval) < 1:
        raise ValueError("eval_interval must be >= 1.")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be > 0.")
    if float(args.threshold) < 0.0:
        raise ValueError("threshold must be >= 0.")
    if float(args.tol) < 0.0:
        raise ValueError("tol must be >= 0.")
    if float(args.scale_init) < 0.0:
        raise ValueError("scale_init must be >= 0.")
    if float(args.bias_scale_init) < 0.0:
        raise ValueError("bias_scale_init must be >= 0.")
    if str(args.optimizer).lower() != "adam":
        raise ValueError("Section 5.1 plan fixes optimizer='adam' for both models.")
    if bool(args.renormalize_projected_probs):
        raise ValueError("Section 5.1 main plan requires renormalize_projected_probs=False.")
    if not bool(args.use_scrambler):
        raise ValueError("Section 5.1 plan fixes use_scrambler=True.")
    if args.aggregate_only and not args.input_result_csvs:
        raise ValueError("--aggregate-only requires --input-result-csvs.")


def _seed_pair_for_instance(*, n_sys: int, M: int, instance_id: int) -> tuple[int, int]:
    benchmark_seed = 100000 * int(n_sys) + 1000 * int(M) + int(instance_id)
    data_seed = 200000 * int(n_sys) + 1000 * int(M) + int(instance_id)
    return int(benchmark_seed), int(data_seed)


def _validate_rows_match_projection_strategy(
    rows: Sequence[dict[str, Any]],
    *,
    expected_strategy: str,
) -> None:
    if not rows:
        return
    normalized_expected = normalize_projection_strategy(expected_strategy)
    normalized_seen = {
        normalize_projection_strategy(str(row["projection_strategy"]))
        for row in rows
        if row.get("projection_strategy") is not None
    }
    if not normalized_seen:
        return
    if normalized_seen != {normalized_expected}:
        raise ValueError(
            "Input rows mix projection strategies or do not match the requested strategy: "
            f"expected={normalized_expected}, seen={sorted(normalized_seen)}."
        )


def _materialize_outputs(
    *,
    rows: Sequence[dict[str, Any]],
    restart_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    m_grid: dict[int, list[int]],
) -> dict[str, Any]:
    _validate_rows_match_projection_strategy(rows, expected_strategy=str(args.projection_strategy))
    dirs = _prepare_output_dirs(output_dir)
    aggregated_rows, regime_summary = aggregate_rows(rows)

    raw_csv_path = dirs["raw"] / "wh_md_sweep_results.csv"
    restart_jsonl_path = dirs["raw"] / "wh_md_sweep_restart_records.jsonl"
    gap_plot_path = dirs["figures"] / "wh_md_sweep_gap_plot.png"
    summary_json_path = dirs["summaries"] / "wh_md_sweep_summary.json"

    _write_results_csv(raw_csv_path, rows)
    _write_restart_jsonl(restart_jsonl_path, restart_rows)
    _plot_gap(aggregated_rows, gap_plot_path, dpi=int(args.plot_dpi))

    if rows:
        simulation_backends = sorted(
            {
                normalize_simulation_backend(row.get("simulation_backend") or "pennylane")
                for row in rows
            }
        )
    else:
        simulation_backends = [normalize_simulation_backend(getattr(args, "simulation_backend", "pennylane"))]
    simulation_backend_summary = simulation_backends[0] if len(simulation_backends) == 1 else "mixed"

    summary = {
        "config": {
            "n_sys_list": [int(value) for value in args.n_sys_list],
            "M_list_by_d": {str(d): [int(value) for value in values] for d, values in m_grid.items()},
            "num_instances_per_grid_point": int(args.num_instances_per_grid_point),
            "instance_ids": [int(value) for value in resolve_instance_ids(args)],
            "num_restarts": int(args.num_restarts),
            "optimizer_name": "adam",
            "learning_rate": float(args.learning_rate),
            "learning_rate_schedule": "constant",
            "steps": int(args.steps),
            "eval_interval": int(args.eval_interval),
            "threshold": float(args.threshold),
            "tol": float(args.tol),
            "trainer": str(args.trainer),
            "loss_type": str(args.loss_type),
            "su_depth": int(args.su_depth),
            "scale_init": float(args.scale_init),
            "bias_scale_init": float(args.bias_scale_init),
            "projection_strategy": str(args.projection_strategy),
            "simulation_backend": simulation_backend_summary,
            "simulation_backends": simulation_backends,
            "project_all_raw_outcomes": bool(
                projection_strategy_projects_all_raw_outcomes(str(args.projection_strategy))
            ),
            "renormalize_projected_probs": bool(args.renormalize_projected_probs),
            "prior_type": "uniform",
            "use_scrambler": bool(args.use_scrambler),
            "aggregate_only": bool(args.aggregate_only),
        },
        "aggregated_by_grid": aggregated_rows,
        "regime_summary": regime_summary,
        "artifacts": {
            "output_dir": str(output_dir),
            "results_csv": str(raw_csv_path),
            "restart_records_jsonl": str(restart_jsonl_path),
            "gap_plot_png": str(gap_plot_path),
            "summary_json": str(summary_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {raw_csv_path}")
    print(f"saved: {restart_jsonl_path}")
    print(f"saved: {gap_plot_path}")
    print(f"saved: {summary_json_path}")
    return summary


def aggregate_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    rows = _load_rows_from_csvs(args.input_result_csvs)
    restart_rows = _load_restart_rows_from_jsonls(args.input_restart_jsonls)
    output_dir = Path(args.output_dir).expanduser().resolve()
    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        args=args,
        output_dir=output_dir,
        m_grid=resolve_m_grid(args),
    )


def run_wh_md_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    m_grid = resolve_m_grid(args)
    instance_ids = resolve_instance_ids(args)

    for n_sys in (int(value) for value in args.n_sys_list):
        d = 2 ** n_sys
        args.n_sys = int(n_sys)
        for M in m_grid[d]:
            for instance_id in instance_ids:
                benchmark_seed, data_seed = _seed_pair_for_instance(n_sys=n_sys, M=M, instance_id=instance_id)
                problem_args = _problem_namespace(
                    n_sys=n_sys,
                    m_outcome=M,
                    benchmark_seed=benchmark_seed,
                    data_seed=data_seed,
                    optimizer="adam",
                    learning_rate=float(args.learning_rate),
                    steps=int(args.steps),
                    eval_interval=int(args.eval_interval),
                    threshold=float(args.threshold),
                    tol=float(args.tol),
                    su_depth=int(args.su_depth),
                    scale_init=float(args.scale_init),
                    bias_scale_init=float(args.bias_scale_init),
                    weight_decay=0.0,
                    state_dtype=str(args.state_dtype),
                )
                problem = _build_problem_instance(problem_args)
                target_states = jnp.arange(int(M), dtype=jnp.int32)
                raw_outcomes = 2 ** int(problem["n_anc"])
                groups, mapping_payload = build_projection_groups(
                    raw_outcomes,
                    M,
                    strategy=str(args.projection_strategy),
                )
                p_opt = float(
                    compute_optimum_success_probability(
                        problem=problem,
                        n_sys=int(n_sys),
                        m_outcome=int(M),
                    )
                )

                model_args = argparse.Namespace(**vars(args))
                model_args.n_sys = int(n_sys)
                model_args.m_outcome = int(M)
                checkpoint_path = build_restart_checkpoint_path(
                    output_dir,
                    n_sys=int(n_sys),
                    M=int(M),
                    instance_id=int(instance_id),
                    model_type=str(FULL_UCR_SPEC.model_type),
                )
                full_summary, model_restart_rows = _run_model_restarts(
                    spec=FULL_UCR_SPEC,
                    problem=problem,
                    args=model_args,
                    groups=groups,
                    target_states=target_states,
                    checkpoint_path=checkpoint_path,
                    instance_id=int(instance_id),
                    benchmark_seed=int(benchmark_seed),
                    data_seed=int(data_seed),
                )
                for restart_row in model_restart_rows:
                    enriched_restart_row = dict(restart_row)
                    enriched_restart_row.update(
                        {
                            "instance_id": int(instance_id),
                            "n_sys": int(n_sys),
                            "d": int(d),
                            "M": int(M),
                            "M_over_d": float(M / d),
                            "benchmark_seed": int(benchmark_seed),
                            "data_seed": int(data_seed),
                        }
                    )
                    restart_rows.append(enriched_restart_row)

                num_ucr_params_full, num_ucr_params_degree1_budget, compression_ratio = compute_ucr_parameter_counts(
                    n_sys=n_sys,
                    n_anc=int(problem["n_anc"]),
                )
                training_config = _training_metadata_config(args)

                p_succ_full = float(full_summary["p_succ"])
                gap_abs_full = float(p_opt - p_succ_full)
                row = {
                    "instance_id": int(instance_id),
                    "n_sys": int(n_sys),
                    "d": int(d),
                    "M": int(M),
                    "M_over_d": float(M / d),
                    "fiducial_id": f"weyl_seed_{benchmark_seed}",
                    "orbit_index_set_id": f"weyl_labels_seed_{data_seed}_unique",
                    "prior_type": "uniform",
                    "benchmark_seed": int(benchmark_seed),
                    "data_seed": int(data_seed),
                    "raw_outcomes": int(raw_outcomes),
                    "effective_m_outcomes": int(M),
                    "projection_strategy": str(mapping_payload["strategy"]),
                    "class_group_sizes": json.dumps(mapping_payload["class_group_sizes"]),
                    "coverage_ratio": float(mapping_payload["coverage_ratio"]),
                    "p_opt": float(p_opt),
                    "p_succ_full": float(p_succ_full),
                    "gap_abs_full": float(gap_abs_full),
                    "gap_rel_full": float(gap_abs_full / max(p_opt, 1e-12)),
                    "num_ucr_params_full": int(num_ucr_params_full),
                    "num_ucr_params_degree1_budget": int(num_ucr_params_degree1_budget),
                    "ucr_compression_ratio": float(compression_ratio),
                    "optimizer_name": training_config["optimizer_name"],
                    "learning_rate": training_config["learning_rate"],
                    "learning_rate_schedule": training_config["learning_rate_schedule"],
                    "max_steps": training_config["max_steps"],
                    "eval_interval": training_config["eval_interval"],
                    "threshold": training_config["threshold"],
                    "num_restarts": training_config["num_restarts"],
                    "stopping_rule": training_config["stopping_rule"],
                    "aggregation_rule": training_config["aggregation_rule"],
                    "best_restart_full": int(full_summary["best_restart"]),
                    "num_steps_full": int(full_summary["num_steps"]),
                    "termination_reason_full": str(full_summary["termination_reason"]),
                    "wall_clock_sec_full": float(full_summary["wall_clock_sec"]),
                    "simulation_backend": normalize_simulation_backend(
                        getattr(args, "simulation_backend", "pennylane")
                    ),
                }
                rows.append(row)
                print(
                    f"[instance] n_sys={n_sys} d={d} M={M} instance_id={instance_id} "
                    f"gap_full={gap_abs_full:.6f}",
                    flush=True,
                )

    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        args=args,
        output_dir=output_dir,
        m_grid=m_grid,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Section 5.1 Weyl-Heisenberg M/d sweep with projected M-outcome "
            "objective, restart aggregation, and PNG figure outputs."
        )
    )
    parser.add_argument("--n-sys-list", type=int, nargs="+", default=list(DEFAULT_N_SYS_LIST))
    parser.add_argument("--m-values", type=int, nargs="+", default=None)
    parser.add_argument("--instance-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-instances-per-grid-point", type=int, default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--num-restarts", type=int, default=DEFAULT_NUM_RESTARTS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--su-depth", type=int, default=DEFAULT_SU_DEPTH)
    parser.add_argument("--scale-init", type=float, default=DEFAULT_SCALE_INIT)
    parser.add_argument("--bias-scale-init", type=float, default=DEFAULT_BIAS_SCALE_INIT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--trainer", type=str, choices=["full", "debug"], default="full")
    parser.add_argument("--loss-type", type=str, choices=["linear", "js", "nll"], default="linear")
    parser.add_argument("--device-name", type=str, default="default.qubit")
    parser.add_argument("--diff-method", type=str, default="backprop")
    parser.add_argument("--jit-backend", type=str, default="gpu")
    parser.add_argument(
        "--simulation-backend",
        type=str,
        choices=["pennylane", "jax_statevector"],
        default="pennylane",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--projection-strategy", type=str, default="balanced_tail_block")
    parser.add_argument(
        "--renormalize-projected-probs",
        type=_parse_bool_arg,
        default=False,
        metavar="{True,False}",
    )
    parser.add_argument("--state-dtype", type=str, choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--use-scrambler", type=_parse_bool_arg, default=True, metavar="{True,False}")
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--input-result-csvs", type=str, nargs="+", default=None)
    parser.add_argument("--input-restart-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.aggregate_only):
        aggregate_existing_outputs(args)
        return
    run_wh_md_sweep(args)


if __name__ == "__main__":
    main()
