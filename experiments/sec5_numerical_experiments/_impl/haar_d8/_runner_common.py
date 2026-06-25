from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

CURRENT_DIR = Path(__file__).resolve().parent
SEC5_DIR = CURRENT_DIR.parent
UCR_METHOD_DIR = SEC5_DIR.parent
RANDOM_SPARSE_DIR = SEC5_DIR / "random_sparse_model"
SRC_DIR = (SEC5_DIR / "../../../src").resolve()
for path in (CURRENT_DIR, SEC5_DIR, UCR_METHOD_DIR, RANDOM_SPARSE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from walsh_ucr.engine import make_batched_qnode
from walsh_ucr.models.vqsd import WalshKLocalVQSD
from walsh_ucr.training.trainer import TrainResult
from walsh_ucr.utils.sdp import sdp_med

from random_sparse_ucr_vs_degree1 import (
    apply_parameter_mask,
    build_random_sparse_ucr_mask,
    make_masked_loss_fns,
)
from wh_d8_sweep import (
    ModelSpec,
    _append_restart_checkpoint_record,
    _build_model,
    _extract_best_objective,
    _load_restart_checkpoint_records,
    _make_restart_checkpoint_record,
    _resume_state_from_restart_checkpoint_records,
    _termination_reason,
    _validate_restart_checkpoint_records,
    make_projected_losses,
)
from restart_reuse import _make_shared_trainer


DUPLICATE_FIDELITY_THRESHOLD = 0.98
def _parse_bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def _make_batched_qnode_for_problem(
    *,
    problem: dict[str, Any],
    model: Any,
    n_sys: int,
    device_name: str,
    diff_method: str,
) -> Any:
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


def _compute_optimum_success_probability(*, states_np: np.ndarray, M: int) -> float:
    states = np.asarray(states_np[: int(M)], dtype=np.complex128)
    q_rhos = [
        (1.0 / float(M)) * np.outer(states[idx], np.conj(states[idx]))
        for idx in range(int(M))
    ]
    sdp_error, _ = sdp_med(q_rhos, int(M), num_povm=int(M))
    return float(1.0 - float(sdp_error))


def _single_qubit_purity(psi: np.ndarray, *, qubit: int, n_sys: int) -> float:
    tensor = np.asarray(psi, dtype=np.complex128).reshape([2] * int(n_sys))
    axes = [int(qubit)] + [axis for axis in range(int(n_sys)) if axis != int(qubit)]
    mat = np.transpose(tensor, axes).reshape(2, -1)
    rho = mat @ np.conj(mat.T)
    return float(np.real(np.trace(rho @ rho)))


def _ensemble_diagnostics(states_np: np.ndarray) -> dict[str, Any]:
    states = np.asarray(states_np, dtype=np.complex128)
    gram = states @ np.conj(states.T)
    fidelity = np.abs(gram) ** 2
    M = int(states.shape[0])
    offdiag = fidelity[np.triu_indices(M, k=1)] if M > 1 else np.asarray([], dtype=np.float64)
    purities = np.asarray(
        [
            _single_qubit_purity(psi, qubit=qubit, n_sys=3)
            for psi in states
            for qubit in range(3)
        ],
        dtype=np.float64,
    )
    if offdiag.size:
        pairwise_mean = float(np.mean(offdiag))
        pairwise_std = float(np.std(offdiag))
        pairwise_max = float(np.max(offdiag))
    else:
        pairwise_mean = 0.0
        pairwise_std = 0.0
        pairwise_max = 0.0
    return {
        "pairwise_fidelity_mean": pairwise_mean,
        "pairwise_fidelity_std": pairwise_std,
        "pairwise_fidelity_max": pairwise_max,
        "pairwise_fidelity_count": int(offdiag.size),
        "frame_potential_2": float(np.mean(np.abs(gram) ** 4)),
        "single_qubit_purity_mean": float(np.mean(purities)),
        "single_qubit_purity_std": float(np.std(purities)),
        "single_qubit_purity_min": float(np.min(purities)),
        "single_qubit_purity_max": float(np.max(purities)),
        "duplicate_fidelity_threshold": float(DUPLICATE_FIDELITY_THRESHOLD),
        "duplicate_threshold_exceeded": bool(pairwise_max > DUPLICATE_FIDELITY_THRESHOLD),
    }


def _walsh_degree1_parameter_count(*, n_sys: int, n_anc: int) -> int:
    return int(
        sum(
            1 + WalshKLocalVQSD.num_k_local_terms(int(n_sys) + block_idx, 1)
            for block_idx in range(int(n_anc))
        )
    )


def _run_model_restarts(
    *,
    spec: ModelSpec,
    problem: dict[str, Any],
    args: argparse.Namespace,
    groups: Sequence[Sequence[int]],
    target_states: jax.Array,
    instance_id: int,
    benchmark_seed: int,
    data_seed: int,
    checkpoint_path: Path | None,
    sparse_seed_offset: int,
    log_prefix: str = "haar",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
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

    mask_payload: dict[str, Any] | None = None
    trainable_mask = None
    fixed_theta = None
    if str(spec.model_type) == "random_sparse_ucr":
        trainable_mask, mask_payload = build_random_sparse_ucr_mask(
            model=model,
            n_sys=n_sys,
            m_outcome=int(args.m_outcome),
            instance_id=int(instance_id),
            sparse_seed_offset=int(sparse_seed_offset),
        )
        fixed_theta = jnp.zeros((int(model.layout.theta_dim),), dtype=jnp.float64)

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
    if trainable_mask is not None and fixed_theta is not None:
        train_loss_fn, eval_loss_fn = make_masked_loss_fns(
            train_loss_fn=train_loss_fn,
            eval_loss_fn=eval_loss_fn,
            trainable_mask=trainable_mask,
            fixed_theta=fixed_theta,
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

    if checkpoint_path is not None:
        checkpoint_records = [
            record
            for record in _load_restart_checkpoint_records(checkpoint_path)
            if int(record["restart_id"]) < int(args.num_restarts)
        ]
        if checkpoint_records:
            _validate_restart_checkpoint_records(
                checkpoint_records,
                spec=spec,
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
            print(
                f"[resume][{log_prefix}] model={spec.model_type} n_sys={args.n_sys} "
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
        p_succ = float(1.0 - final_objective)
        theta_for_checkpoint = result.theta
        if trainable_mask is not None and fixed_theta is not None:
            theta_for_checkpoint = apply_parameter_mask(result.theta, trainable_mask, fixed_theta)

        record = {
            "model_type": str(spec.model_type),
            "model_name": str(spec.model_name),
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
                theta=theta_for_checkpoint,
                spec=spec,
                n_sys=n_sys,
                M=int(args.m_outcome),
                instance_id=int(instance_id),
                benchmark_seed=int(benchmark_seed),
                data_seed=int(data_seed),
                args=args,
            )
            _append_restart_checkpoint_record(checkpoint_path, checkpoint_record)
            print(
                f"[restart][checkpointed][{log_prefix}] model={spec.model_type} "
                f"n_sys={args.n_sys} M={args.m_outcome} instance_id={instance_id} "
                f"restart_id={restart_id}",
                flush=True,
            )

        if best_record is None or final_objective < float(best_record["final_objective_value"]):
            best_record = record
            best_theta = theta_for_checkpoint

    if best_record is None or best_theta is None:
        raise RuntimeError(f"No valid restart result found for model '{spec.model_type}'.")

    return (
        {
            "model_type": str(spec.model_type),
            "model_name": str(spec.model_name),
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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_csv_rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _load_jsonl_rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _enrich_restart_rows(
    rows: Sequence[dict[str, Any]],
    *,
    n_sys: int,
    d: int,
    M: int,
    instance_id: int,
    M_over_d: float,
    benchmark_seed: int,
    data_seed: int,
    ensemble_seed: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "n_sys": int(n_sys),
                "d": int(d),
                "M": int(M),
                "M_over_d": float(M_over_d),
                "instance_id": int(instance_id),
                "benchmark_seed": int(benchmark_seed),
                "data_seed": int(data_seed),
                "ensemble_seed": int(ensemble_seed),
            }
        )
        enriched.append(item)
    return enriched


def _model_result_fields(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        f"p_succ_{prefix}": float(summary["p_succ"]),
        f"best_restart_{prefix}": int(summary["best_restart"]),
        f"seed_opt_{prefix}": int(summary["seed_opt"]),
        f"num_steps_{prefix}": int(summary["num_steps"]),
        f"termination_reason_{prefix}": str(summary["termination_reason"]),
        f"wall_clock_sec_{prefix}": float(summary["wall_clock_sec"]),
    }
