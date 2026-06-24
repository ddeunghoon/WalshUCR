from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
UCR_METHOD_DIR = CURRENT_DIR.parent
SRC_DIR = (CURRENT_DIR / "../../../src").resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(UCR_METHOD_DIR) not in sys.path:
    sys.path.append(str(UCR_METHOD_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scalable_vqsd.models.vqsd import WalshKLocalVQSD

from atucr_weyl_init_sweep import _build_problem_instance, _parse_bool_arg
from wh_md_sweep import (
    ModelSpec,
    _problem_namespace,
    _seed_pair_for_instance,
    _stats_payload,
    _training_metadata_config,
    _validate_args,
    build_projection_groups,
    build_restart_checkpoint_path,
    compute_optimum_success_probability,
    normalize_projection_strategy,
    resolve_instance_ids,
    resolve_m_grid,
)
from wh_md_sweep_restart_reuse import _run_model_restarts_restart_reuse


DEFAULT_OUTPUT_ROOT = (
    CURRENT_DIR
    / "results"
    / "wh_md_walsh_degree1_nsys3_scale1_drop_extra_restart_reuse_i10_r50"
)
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

WALSH_DEGREE1_SPEC = ModelSpec(
    model_type="walsh_degree_1",
    model_name="walsh_k_local",
    mean_init="0.0",
    bias_mean_init="pi/2",
    ucr_degree=1,
)

RESULTS_FILENAME = "wh_md_walsh_degree1_results.csv"
RESTARTS_FILENAME = "wh_md_walsh_degree1_restart_records.jsonl"
SUMMARY_FILENAME = "wh_md_walsh_degree1_summary.json"
GAP_PLOT_FILENAME = "wh_md_walsh_degree1_gap_left_panel.png"

INT_FIELDS = {
    "instance_id",
    "n_sys",
    "d",
    "M",
    "benchmark_seed",
    "data_seed",
    "raw_outcomes",
    "effective_m_outcomes",
    "num_ucr_params_walsh_deg1",
    "best_restart_walsh_deg1",
    "seed_opt_walsh_deg1",
    "num_steps_walsh_deg1",
    "num_restarts",
    "max_steps",
    "eval_interval",
}
FLOAT_FIELDS = {
    "M_over_d",
    "coverage_ratio",
    "p_opt",
    "p_succ_walsh_deg1",
    "gap_abs_walsh_deg1",
    "gap_rel_walsh_deg1",
    "learning_rate",
    "threshold",
    "wall_clock_sec_walsh_deg1",
}


def walsh_degree1_parameter_count(*, n_sys: int, n_anc: int) -> int:
    return int(
        sum(
            1 + WalshKLocalVQSD.num_k_local_terms(int(n_sys) + block_idx, 1)
            for block_idx in range(int(n_anc))
        )
    )


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "figures": figures_dir, "summaries": summaries_dir}


def _result_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
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
        "model_type",
        "model_name",
        "walsh_degree",
        "p_opt",
        "p_succ_walsh_deg1",
        "gap_abs_walsh_deg1",
        "gap_rel_walsh_deg1",
        "num_ucr_params_walsh_deg1",
        "optimizer_name",
        "learning_rate",
        "learning_rate_schedule",
        "max_steps",
        "eval_interval",
        "threshold",
        "num_restarts",
        "stopping_rule",
        "aggregation_rule",
        "best_restart_walsh_deg1",
        "seed_opt_walsh_deg1",
        "num_steps_walsh_deg1",
        "termination_reason_walsh_deg1",
        "wall_clock_sec_walsh_deg1",
    ]
    extra = sorted({str(key) for row in rows for key in row if str(key) not in preferred})
    return preferred + extra


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = _result_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_restart_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _coerce_csv_value(key: str, value: str) -> Any:
    if key in INT_FIELDS and value != "":
        return int(value)
    if key in FLOAT_FIELDS and value != "":
        return float(value)
    return value


def _read_result_csvs(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(
                {key: _coerce_csv_value(key, value) for key, value in row.items()}
                for row in reader
            )
    return rows


def _read_restart_jsonls(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["n_sys"]), int(row["M"])), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        first = bucket[0]
        payload: dict[str, Any] = {
            "n_sys": int(first["n_sys"]),
            "d": int(first["d"]),
            "M": int(first["M"]),
            "M_over_d": float(first["M_over_d"]),
            "count": int(len(bucket)),
        }
        for column in ("gap_abs_walsh_deg1", "gap_rel_walsh_deg1", "p_succ_walsh_deg1"):
            stats = _stats_payload([float(row[column]) for row in bucket])
            for stat_name, stat_value in stats.items():
                payload[f"{column}_{stat_name}"] = stat_value
        aggregated.append(payload)
    return aggregated


def _instance_offsets(rows: Sequence[dict[str, Any]]) -> dict[int, float]:
    offsets = [-0.038, -0.019, 0.0, 0.019, 0.038]
    return {
        instance_id: offsets[idx % len(offsets)]
        for idx, instance_id in enumerate(sorted({int(row["instance_id"]) for row in rows}))
    }


def _plot_gap_left_panel(
    rows: Sequence[dict[str, Any]],
    aggregated_rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    dpi: int,
    display_floor: float,
) -> None:
    if not rows:
        return

    m_values = sorted({int(row["M"]) for row in rows})
    instance_offset_by_id = _instance_offsets(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    scatter_x = [
        int(row["M"]) + instance_offset_by_id[int(row["instance_id"])]
        for row in rows
    ]
    scatter_y = [
        max(float(row["gap_abs_walsh_deg1"]), float(display_floor))
        for row in rows
    ]
    ax.scatter(scatter_x, scatter_y, s=13, color="tab:purple", marker="D", alpha=0.48, linewidths=0.0)

    sorted_agg = sorted(aggregated_rows, key=lambda row: int(row["M"]))
    ax.plot(
        [int(row["M"]) for row in sorted_agg],
        [max(float(row["gap_abs_walsh_deg1_median"]), float(display_floor)) for row in sorted_agg],
        color="tab:purple",
        marker="D",
        markersize=3.5,
        linewidth=1.25,
        label="Walsh degree-1",
    )

    ax.set_xlabel("M")
    ax.set_ylabel("Absolute optimum gap")
    ax.set_xticks(m_values)
    ax.set_xlim(min(m_values) - 0.55, max(m_values) + 0.55)
    ax.set_yscale("log")
    ax.set_ylim(float(display_floor), 7e-2)
    ax.axhline(float(display_floor), color="0.35", linewidth=0.6, linestyle=":")
    ax.grid(True, which="major", axis="both", color="0.85", linewidth=0.55)
    ax.grid(True, which="minor", axis="y", color="0.92", linewidth=0.35)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _materialize_outputs(
    *,
    rows: Sequence[dict[str, Any]],
    restart_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    m_grid: dict[int, list[int]],
) -> dict[str, Any]:
    dirs = _prepare_output_dirs(output_dir)
    aggregated_rows = aggregate_rows(rows)

    raw_csv_path = dirs["raw"] / RESULTS_FILENAME
    restart_jsonl_path = dirs["raw"] / RESTARTS_FILENAME
    gap_plot_path = dirs["figures"] / GAP_PLOT_FILENAME
    summary_json_path = dirs["summaries"] / SUMMARY_FILENAME

    _write_results_csv(raw_csv_path, rows)
    _write_restart_jsonl(restart_jsonl_path, restart_rows)
    _plot_gap_left_panel(
        rows,
        aggregated_rows,
        gap_plot_path,
        dpi=int(args.plot_dpi),
        display_floor=float(args.display_floor),
    )

    summary = {
        "config": {
            "n_sys_list": [int(value) for value in args.n_sys_list],
            "M_list_by_d": {str(d): [int(value) for value in values] for d, values in m_grid.items()},
            "instance_ids": [int(value) for value in resolve_instance_ids(args)],
            "num_instances_per_grid_point": int(args.num_instances_per_grid_point),
            "num_restarts": int(args.num_restarts),
            "seed_start": int(args.seed_start),
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
            "projection_strategy": normalize_projection_strategy(str(args.projection_strategy)),
            "renormalize_projected_probs": bool(args.renormalize_projected_probs),
            "prior_type": "uniform",
            "use_scrambler": bool(args.use_scrambler),
            "model_type": str(WALSH_DEGREE1_SPEC.model_type),
            "model_name": str(WALSH_DEGREE1_SPEC.model_name),
            "walsh_degree": int(WALSH_DEGREE1_SPEC.ucr_degree or 1),
            "display_floor": float(args.display_floor),
            "aggregate_only": bool(args.aggregate_only),
        },
        "aggregated_by_grid": aggregated_rows,
        "artifacts": {
            "output_dir": str(output_dir),
            "results_csv": str(raw_csv_path),
            "restart_records_jsonl": str(restart_jsonl_path),
            "gap_left_panel_png": str(gap_plot_path),
            "summary_json": str(summary_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {raw_csv_path}")
    print(f"saved: {restart_jsonl_path}")
    print(f"saved: {gap_plot_path}")
    print(f"saved: {summary_json_path}")
    return summary


def run_walsh_degree1_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    m_grid = resolve_m_grid(args)
    instance_ids = resolve_instance_ids(args)

    for n_sys in (int(value) for value in args.n_sys_list):
        d = 2 ** int(n_sys)
        args.n_sys = int(n_sys)
        for M in m_grid[d]:
            for instance_id in instance_ids:
                benchmark_seed, data_seed = _seed_pair_for_instance(n_sys=n_sys, M=M, instance_id=instance_id)
                problem_args = _problem_namespace(
                    n_sys=int(n_sys),
                    m_outcome=int(M),
                    benchmark_seed=int(benchmark_seed),
                    data_seed=int(data_seed),
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
                    int(M),
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
                    model_type=str(WALSH_DEGREE1_SPEC.model_type),
                )
                summary, model_restart_rows = _run_model_restarts_restart_reuse(
                    spec=WALSH_DEGREE1_SPEC,
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
                    enriched = dict(restart_row)
                    enriched.update(
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
                    restart_rows.append(enriched)

                training_config = _training_metadata_config(args)
                p_succ = float(summary["p_succ"])
                gap_abs = float(p_opt - p_succ)
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
                    "model_type": str(WALSH_DEGREE1_SPEC.model_type),
                    "model_name": str(WALSH_DEGREE1_SPEC.model_name),
                    "walsh_degree": int(WALSH_DEGREE1_SPEC.ucr_degree or 1),
                    "p_opt": float(p_opt),
                    "p_succ_walsh_deg1": float(p_succ),
                    "gap_abs_walsh_deg1": float(gap_abs),
                    "gap_rel_walsh_deg1": float(gap_abs / max(p_opt, 1e-12)),
                    "num_ucr_params_walsh_deg1": walsh_degree1_parameter_count(
                        n_sys=int(n_sys),
                        n_anc=int(problem["n_anc"]),
                    ),
                    "optimizer_name": training_config["optimizer_name"],
                    "learning_rate": training_config["learning_rate"],
                    "learning_rate_schedule": training_config["learning_rate_schedule"],
                    "max_steps": training_config["max_steps"],
                    "eval_interval": training_config["eval_interval"],
                    "threshold": training_config["threshold"],
                    "num_restarts": training_config["num_restarts"],
                    "stopping_rule": training_config["stopping_rule"],
                    "aggregation_rule": training_config["aggregation_rule"],
                    "best_restart_walsh_deg1": int(summary["best_restart"]),
                    "seed_opt_walsh_deg1": int(summary["seed_opt"]),
                    "num_steps_walsh_deg1": int(summary["num_steps"]),
                    "termination_reason_walsh_deg1": str(summary["termination_reason"]),
                    "wall_clock_sec_walsh_deg1": float(summary["wall_clock_sec"]),
                }
                rows.append(row)
                print(
                    f"[walsh-degree1] n_sys={n_sys} d={d} M={M} instance_id={instance_id} "
                    f"gap={gap_abs:.6f} p_succ={p_succ:.6f}",
                    flush=True,
                )

    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        args=args,
        output_dir=output_dir,
        m_grid=m_grid,
    )


def aggregate_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    rows = _read_result_csvs(args.input_result_csvs)
    restart_rows = _read_restart_jsonls(args.input_restart_jsonls)
    return _materialize_outputs(
        rows=rows,
        restart_rows=restart_rows,
        args=args,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        m_grid=resolve_m_grid(args),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Walsh degree-1 only WH M/d sweep matching the Section 5 left-panel grid."
    )
    parser.add_argument("--n-sys-list", type=int, nargs="+", default=list(DEFAULT_N_SYS_LIST))
    parser.add_argument("--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    parser.add_argument("--instance-ids", type=int, nargs="+", default=list(DEFAULT_INSTANCE_IDS))
    parser.add_argument("--num-instances-per-grid-point", type=int, default=len(DEFAULT_INSTANCE_IDS))
    parser.add_argument("--num-restarts", type=int, default=DEFAULT_NUM_RESTARTS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--su-depth", type=int, default=DEFAULT_SU_DEPTH)
    parser.add_argument("--scale-init", type=float, default=DEFAULT_SCALE_INIT)
    parser.add_argument("--bias-scale-init", type=float, default=DEFAULT_BIAS_SCALE_INIT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--trainer", type=str, choices=["full"], default="full")
    parser.add_argument("--loss-type", type=str, choices=["linear"], default="linear")
    parser.add_argument("--device-name", type=str, default="default.qubit")
    parser.add_argument("--diff-method", type=str, default="backprop")
    parser.add_argument("--jit-backend", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--projection-strategy", type=str, default="drop_extra")
    parser.add_argument(
        "--renormalize-projected-probs",
        type=_parse_bool_arg,
        default=False,
        metavar="{True,False}",
    )
    parser.add_argument("--state-dtype", type=str, choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--use-scrambler", type=_parse_bool_arg, default=True, metavar="{True,False}")
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--display-floor", type=float, default=DEFAULT_DISPLAY_FLOOR)
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
    run_walsh_degree1_sweep(args)


if __name__ == "__main__":
    main()
