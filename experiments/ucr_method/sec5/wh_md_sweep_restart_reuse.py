from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from scalable_vqsd.training.trainer import JAX_Full_Trainer, TrainResult

from wh_md_sweep import (
    _append_restart_checkpoint_record,
    _build_batched_qnode_for_problem,
    _build_model,
    _extract_best_objective,
    _load_restart_checkpoint_records,
    _make_restart_checkpoint_record,
    _resume_state_from_restart_checkpoint_records,
    _termination_reason,
    _validate_restart_checkpoint_records,
    make_projected_losses,
    normalize_simulation_backend,
)


def _make_shared_trainer(
    *,
    train_loss_fn: Any,
    eval_loss_fn: Any,
    theta_template: Any,
    m_outcome: int,
    learning_rate: float,
    eval_interval: int,
) -> JAX_Full_Trainer:
    a_priori_probs = jnp.ones((int(m_outcome),), dtype=jnp.float64) / float(m_outcome)
    return JAX_Full_Trainer(
        train_cost_fn=train_loss_fn,
        theta_init=theta_template,
        optimizer_name="adam",
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        eval_interval=int(eval_interval),
        eval_cost_fn=eval_loss_fn,
        n_outcome=int(m_outcome),
        a_priori_probs=a_priori_probs,
    )


def _run_model_restarts_restart_reuse(
    *,
    spec,
    problem: dict[str, Any],
    args: argparse.Namespace,
    groups,
    target_states: jax.Array,
    checkpoint_path: Path | None = None,
    instance_id: int | None = None,
    benchmark_seed: int | None = None,
    data_seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if str(args.trainer) != "full":
        raise ValueError("restart_reuse execution path supports trainer='full' only.")
    if str(args.optimizer).lower() != "adam":
        raise ValueError("restart_reuse execution path supports optimizer='adam' only.")

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

    theta_template = model.layout.init_params(jax.random.PRNGKey(int(args.seed_start)))
    trainer = _make_shared_trainer(
        train_loss_fn=train_loss_fn,
        eval_loss_fn=eval_loss_fn,
        theta_template=theta_template,
        m_outcome=int(args.m_outcome),
        learning_rate=float(args.learning_rate),
        eval_interval=int(args.eval_interval),
    )

    best_record: dict[str, Any] | None = None
    best_theta: jax.Array | None = None
    restart_records: list[dict[str, Any]] = []
    completed_restart_ids: set[int] = set()
    train_args = (problem["inputs"], target_states)
    eval_args = (problem["inputs"], target_states)

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
                f"[resume][restart_reuse] model={spec.model_type} n_sys={args.n_sys} "
                f"M={args.m_outcome} instance_id={instance_id} "
                f"completed_restarts={len(completed_restart_ids)}",
                flush=True,
            )

    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_restart_ids:
            continue
        seed_opt = int(args.seed_start) + restart_id
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
                f"[restart][checkpointed][restart_reuse] model={spec.model_type} "
                f"n_sys={args.n_sys} M={args.m_outcome} instance_id={instance_id} "
                f"restart_id={restart_id}",
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
