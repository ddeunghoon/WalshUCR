from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp

CURRENT_DIR = Path(__file__).resolve().parent
UCR_METHOD_DIR = CURRENT_DIR.parent
SRC_DIR = (CURRENT_DIR / "../../../src").resolve()
_sys = __import__("sys")
if str(CURRENT_DIR) not in _sys.path:
    _sys.path.append(str(CURRENT_DIR))
if str(UCR_METHOD_DIR) not in _sys.path:
    _sys.path.append(str(UCR_METHOD_DIR))
if str(SRC_DIR) not in _sys.path:
    _sys.path.append(str(SRC_DIR))

from walsh_ucr.training.trainer import JAX_Full_Trainer, TrainResult

from wh_d8_sweep import (
    FULL_UCR_SPEC,
    _append_restart_checkpoint_record,
    _build_batched_qnode_for_problem,
    _build_model,
    _extract_best_objective,
    _make_restart_checkpoint_record,
    _materialize_outputs,
    _problem_namespace,
    _resume_state_from_restart_checkpoint_records,
    _seed_pair_for_instance,
    _termination_reason,
    _training_metadata_config,
    _validate_restart_checkpoint_records,
    _validate_args,
    build_restart_checkpoint_path,
    build_projection_groups,
    compute_optimum_success_probability,
    compute_ucr_parameter_counts,
    make_projected_losses,
    normalize_simulation_backend,
    parse_args,
    _load_restart_checkpoint_records,
    resolve_instance_ids,
    resolve_m_grid,
)
from weyl_problem import _build_problem_instance


def _make_shared_trainer(
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
        raise ValueError("restart_reuse execution path currently supports trainer='full' only.")
    if str(args.optimizer).lower() != "adam":
        raise ValueError("restart_reuse execution path currently supports optimizer='adam' only.")

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
        jit_backend=getattr(args, "jit_backend", None),
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


def run_wh_md_sweep_restart_reuse(args: argparse.Namespace) -> dict[str, Any]:
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
                full_summary, model_restart_rows = _run_model_restarts_restart_reuse(
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
                    "class_group_sizes": __import__("json").dumps(mapping_payload["class_group_sizes"]),
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
                }
                rows.append(row)
                print(
                    f"[instance][restart_reuse] n_sys={n_sys} d={d} M={M} instance_id={instance_id} "
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.aggregate_only):
        from wh_d8_sweep import aggregate_existing_outputs

        aggregate_existing_outputs(args)
        return
    run_wh_md_sweep_restart_reuse(args)


if __name__ == "__main__":
    main()
