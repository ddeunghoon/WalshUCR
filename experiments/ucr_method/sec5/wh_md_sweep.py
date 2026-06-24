from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from scalable_vqsd.engine import make_batched_qnode
from scalable_vqsd.models.vqsd import WalshKLocalVQSD

from atucr_weyl_init_sweep import _compute_sdp_value


jax.config.update("jax_enable_x64", True)

DEFAULT_SEED_START = 0
RESTART_CHECKPOINT_VERSION = 1
PROJECTION_STRATEGY_ALIASES = {
    "drop_extra": "drop_extra",
}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_name: str
    mean_init: str
    bias_mean_init: str | None
    ucr_degree: int | None = None


def normalize_simulation_backend(value: str | None) -> str:
    backend = "pennylane" if value is None else str(value).strip().lower()
    if backend != "pennylane":
        raise ValueError("WalshUCR Section 5 release supports simulation_backend='pennylane' only.")
    return backend


def normalize_projection_strategy(value: str) -> str:
    normalized = str(value).strip().lower()
    try:
        return PROJECTION_STRATEGY_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PROJECTION_STRATEGY_ALIASES))
        raise ValueError(f"Unsupported projection_strategy '{value}'. Supported values: {supported}.") from exc


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
    return groups, {
        "strategy": "drop_extra",
        "raw_outcomes": raw,
        "m_outcomes": m,
        "class_to_outcomes": [group.tolist() for group in groups],
        "class_group_sizes": [1] * m,
        "assigned_raw_outcomes": list(range(m)),
        "unassigned_raw_outcomes": list(range(m, raw)),
        "coverage_ratio": float(m / float(raw)),
    }


def build_projection_groups(
    raw_outcomes: int,
    m_outcomes: int,
    strategy: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    resolved = normalize_projection_strategy(strategy)
    if resolved == "drop_extra":
        return build_drop_extra_groups(raw_outcomes, m_outcomes)
    raise ValueError(f"Unsupported normalized projection strategy '{resolved}'.")


def make_projector(groups: Sequence[np.ndarray]) -> Callable[[jax.Array], jax.Array]:
    group_idx = tuple(jnp.asarray(group, dtype=jnp.int32) for group in groups)

    def projector(raw_probs: jax.Array) -> jax.Array:
        cols = [jnp.sum(raw_probs[:, idx], axis=1) for idx in group_idx]
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

    if str(loss_type) != "linear":
        raise ValueError("WalshUCR Section 5 release supports loss_type='linear' only.")
    return train_loss_linear, forward_eval, projector


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
        n_sys=int(n_sys),
        problem_type="weyl",
        m_outcome=int(m_outcome),
        canonical_labels=False,
        k_inits=1,
        seed_start=DEFAULT_SEED_START,
        benchmark_seed=int(benchmark_seed),
        data_seed=int(data_seed),
        state_dtype=str(state_dtype),
        su_depth=int(su_depth),
        model="walsh_k_local",
        mean_init="0.0",
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
        jit_backend="cpu",
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
) -> WalshKLocalVQSD:
    if spec.model_name != "walsh_k_local" or spec.ucr_degree is None:
        raise ValueError("WalshUCR Section 5 release supports only model_name='walsh_k_local'.")
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


def _build_batched_qnode_for_problem(
    *,
    problem: dict[str, Any],
    model: WalshKLocalVQSD,
    n_sys: int,
    device_name: str,
    diff_method: str,
    simulation_backend: str = "pennylane",
) -> Callable[[Any, jax.Array], jax.Array]:
    if str(problem["problem_type"]) != "weyl":
        raise ValueError("WalshUCR Section 5 release supports Weyl problems only.")
    normalize_simulation_backend(simulation_backend)
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


def compute_optimum_success_probability(*, problem: dict[str, Any], n_sys: int, m_outcome: int) -> float:
    sdp_error = float(_compute_sdp_value(problem=problem, n_sys=int(n_sys), m_outcome=int(m_outcome)))
    return float(1.0 - sdp_error)


def _termination_reason(result: Any, *, max_steps: int) -> str:
    if bool(result.nan_found):
        return "nan"
    if bool(result.stopped_early):
        return "threshold"
    if int(result.steps_run) >= int(max_steps):
        return "max_steps"
    return "unknown"


def _extract_best_objective(loss_log: Any) -> float:
    arr = np.asarray(loss_log, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
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


def resolve_m_grid(args: argparse.Namespace) -> dict[int, list[int]]:
    if not args.m_values:
        raise ValueError("--m-values is required in the WalshUCR Section 5 release script.")
    m_values = [int(value) for value in args.m_values]
    if any(value < 2 for value in m_values):
        raise ValueError(f"All m_values must be >= 2, got {m_values}.")
    return {2 ** int(n_sys): list(m_values) for n_sys in args.n_sys_list}


def resolve_instance_ids(args: argparse.Namespace) -> list[int]:
    if args.instance_ids:
        instance_ids = sorted({int(value) for value in args.instance_ids})
        if any(value < 0 for value in instance_ids):
            raise ValueError(f"instance_ids must be >= 0, got {instance_ids}.")
        return instance_ids
    return list(range(int(args.num_instances_per_grid_point)))


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
        raise ValueError("WalshUCR Section 5 release fixes optimizer='adam'.")
    if str(args.loss_type).lower() != "linear":
        raise ValueError("WalshUCR Section 5 release fixes loss_type='linear'.")
    if bool(args.renormalize_projected_probs):
        raise ValueError("WalshUCR Section 5 release requires renormalize_projected_probs=False.")
    if not bool(args.use_scrambler):
        raise ValueError("WalshUCR Section 5 release requires use_scrambler=True.")
    if args.aggregate_only and not args.input_result_csvs:
        raise ValueError("--aggregate-only requires --input-result-csvs.")


def _seed_pair_for_instance(*, n_sys: int, M: int, instance_id: int) -> tuple[int, int]:
    benchmark_seed = 100000 * int(n_sys) + 1000 * int(M) + int(instance_id)
    data_seed = 200000 * int(n_sys) + 1000 * int(M) + int(instance_id)
    return int(benchmark_seed), int(data_seed)


def build_restart_checkpoint_path(
    output_dir: Path,
    *,
    n_sys: int,
    M: int,
    instance_id: int,
    model_type: str,
) -> Path:
    checkpoint_dir = output_dir / "raw" / "restart_checkpoints"
    return checkpoint_dir / f"nsys{int(n_sys)}_M{int(M)}_instance{int(instance_id):02d}_{str(model_type)}.jsonl"


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
                print(f"warning: ignoring malformed checkpoint line {lineno} in {path}", file=sys.stderr, flush=True)
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
                raise ValueError(f"Checkpoint mismatch for {key}: expected {expected_value}, got {record.get(key)}.")


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
