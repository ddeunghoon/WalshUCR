from __future__ import annotations

import argparse
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
ROOT = CURRENT_DIR.parents[2]
SRC_DIR = (CURRENT_DIR / "../../../src").resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(UCR_METHOD_DIR) not in sys.path:
    sys.path.append(str(UCR_METHOD_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from walsh_ucr.models.vqsd import WalshKLocalVQSD

from common import (
    _fieldnames_with_extras,
    _load_jsonl_rows,
    _prepare_output_dirs,
    _read_rows,
    _read_typed_csv_rows,
    _write_dict_csv,
    _write_jsonl,
)
from weyl_problem import _build_problem_instance, _parse_bool_arg
from wh_d8_sweep import (
    ModelSpec,
    _problem_namespace,
    _seed_pair_for_instance,
    build_projection_groups,
    build_restart_checkpoint_path,
    normalize_projection_strategy,
)
from restart_reuse import _run_model_restarts_restart_reuse


DEFAULT_REFERENCE_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_sweep_results.csv"
)
DEFAULT_INSTANCE_IDS = (0, 1, 2, 3, 4)
DEFAULT_DEGREES = (1, 2, 3, 4, 5)
DEFAULT_M_VALUES = (9, 10, 11, 12)
DEFAULT_N_SYS = 3
DEFAULT_NUM_RESTARTS = 100
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
DEFAULT_NUMERICAL_FLOOR = 3e-5

RESULTS_FILENAME = "wh_md_walsh_degree_sweep_results.csv"
RESTARTS_FILENAME = "wh_md_walsh_degree_sweep_restart_records.jsonl"
BY_M_DEGREE_FILENAME = "wh_md_walsh_degree_sweep_by_m_degree.csv"
BY_INSTANCE_PLOT_FILENAME = "wh_md_walsh_degree_gap_by_instance.png"
MEAN_PLOT_FILENAME = "wh_md_walsh_degree_gap_mean.png"
MEDIAN_IQR_PLOT_FILENAME = "wh_md_walsh_degree_gap_median_iqr.png"
SUMMARY_FILENAME = "wh_md_walsh_degree_sweep_summary.json"

INT_FIELDS = {
    "instance_id",
    "n_sys",
    "d",
    "M",
    "degree",
    "benchmark_seed",
    "data_seed",
    "raw_outcomes",
    "effective_m_outcomes",
    "num_ucr_params_degree",
    "num_ucr_params_full",
    "best_restart",
    "seed_opt",
    "num_steps",
    "num_restarts",
    "seed_start",
    "max_steps",
    "eval_interval",
}
FLOAT_FIELDS = {
    "M_over_d",
    "coverage_ratio",
    "p_opt",
    "p_succ",
    "gap_abs",
    "gap_rel",
    "p_succ_full_ref",
    "gap_abs_full_ref",
    "gap_rel_full_ref",
    "learning_rate",
    "threshold",
    "wall_clock_sec",
}


def default_output_root(
    *,
    n_sys: int,
    m_values: Sequence[int],
    projection_strategy: str,
    instance_ids: Sequence[int],
    num_restarts: int,
) -> Path:
    strategy = normalize_projection_strategy(str(projection_strategy))
    instance_count = len({int(value) for value in instance_ids})
    m_values_int = [int(value) for value in m_values]
    return (
        CURRENT_DIR
        / "results"
        / (
            f"wh_md_walsh_degree_sweep_nsys{int(n_sys)}_M{min(m_values_int)}_M{max(m_values_int)}_"
            f"{strategy}_random_i{int(instance_count)}_r{int(num_restarts)}"
        )
    )


def walsh_degree_model_spec(degree: int) -> ModelSpec:
    degree_int = int(degree)
    return ModelSpec(
        model_type=f"walsh_degree_{degree_int}",
        model_name="walsh_k_local",
        mean_init="0.0",
        bias_mean_init="pi/2",
        ucr_degree=degree_int,
    )


def compute_walsh_k_local_parameter_count(*, n_sys: int, n_anc: int, degree: int) -> int:
    return int(
        sum(
            1 + WalshKLocalVQSD.num_k_local_terms(int(n_sys) + block_idx, int(degree))
            for block_idx in range(int(n_anc))
        )
    )


def _int(row: dict[str, Any], key: str) -> int:
    return int(row[key])


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def load_reference_rows(
    path: Path,
    *,
    n_sys: int,
    m_values: Sequence[int],
    instance_ids: Sequence[int],
) -> list[dict[str, Any]]:
    wanted_m = {int(value) for value in m_values}
    wanted_instances = {int(value) for value in instance_ids}
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    for row in _read_rows(Path(path)):
        if int(row["n_sys"]) != int(n_sys):
            continue
        m_value = int(row["M"])
        instance_id = int(row["instance_id"])
        if m_value not in wanted_m or instance_id not in wanted_instances:
            continue
        key = (m_value, instance_id)
        if key in rows_by_key:
            raise ValueError(f"Duplicate reference row for M={m_value} instance_id={instance_id}.")
        rows_by_key[key] = dict(row)

    missing = [
        (m_value, instance_id)
        for m_value in sorted(wanted_m)
        for instance_id in sorted(wanted_instances)
        if (m_value, instance_id) not in rows_by_key
    ]
    if missing:
        preview = missing[:10]
        raise ValueError(f"Missing reference rows for first keys={preview}.")
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def _trained_degree_row(
    *,
    reference_row: dict[str, Any],
    degree: int,
    summary: dict[str, Any],
    args: argparse.Namespace,
    raw_outcomes: int,
    mapping_payload: dict[str, Any],
) -> dict[str, Any]:
    p_opt = _float(reference_row, "p_opt")
    p_succ = float(summary["p_succ"])
    gap_abs = float(p_opt - p_succ)
    n_anc = int(round(math.log2(int(raw_outcomes))))
    return {
        "instance_id": _int(reference_row, "instance_id"),
        "n_sys": int(args.n_sys),
        "d": int(2 ** int(args.n_sys)),
        "M": _int(reference_row, "M"),
        "M_over_d": float(_int(reference_row, "M") / float(2 ** int(args.n_sys))),
        "degree": int(degree),
        "model_type": f"walsh_degree_{int(degree)}",
        "model_name": "walsh_k_local",
        "source": "trained",
        "p_opt": float(p_opt),
        "p_succ": float(p_succ),
        "gap_abs": float(gap_abs),
        "gap_rel": float(gap_abs / max(p_opt, 1e-12)),
        "p_succ_full_ref": _float(reference_row, "p_succ_full"),
        "gap_abs_full_ref": _float(reference_row, "gap_abs_full"),
        "gap_rel_full_ref": _float(reference_row, "gap_rel_full"),
        "num_ucr_params_degree": compute_walsh_k_local_parameter_count(
            n_sys=int(args.n_sys),
            n_anc=n_anc,
            degree=int(degree),
        ),
        "num_ucr_params_full": _int(reference_row, "num_ucr_params_full"),
        "best_restart": int(summary["best_restart"]),
        "seed_opt": int(summary["seed_opt"]),
        "num_steps": int(summary["num_steps"]),
        "termination_reason": str(summary["termination_reason"]),
        "wall_clock_sec": float(summary["wall_clock_sec"]),
        "optimizer_name": "adam",
        "learning_rate": float(args.learning_rate),
        "learning_rate_schedule": "constant",
        "max_steps": int(args.steps),
        "eval_interval": int(args.eval_interval),
        "threshold": float(args.threshold),
        "num_restarts": int(args.num_restarts),
        "seed_start": int(args.seed_start),
        "projection_strategy": str(mapping_payload["strategy"]),
        "benchmark_seed": _int(reference_row, "benchmark_seed"),
        "data_seed": _int(reference_row, "data_seed"),
        "raw_outcomes": int(raw_outcomes),
        "effective_m_outcomes": _int(reference_row, "M"),
        "coverage_ratio": float(mapping_payload["coverage_ratio"]),
        "class_group_sizes": json.dumps(mapping_payload["class_group_sizes"]),
        "prior_type": "uniform",
        "fiducial_id": str(reference_row["fiducial_id"]),
        "orbit_index_set_id": str(reference_row["orbit_index_set_id"]),
    }


def _results_fieldnames() -> list[str]:
    return [
        "instance_id",
        "n_sys",
        "d",
        "M",
        "M_over_d",
        "degree",
        "model_type",
        "model_name",
        "source",
        "p_opt",
        "p_succ",
        "gap_abs",
        "gap_rel",
        "p_succ_full_ref",
        "gap_abs_full_ref",
        "gap_rel_full_ref",
        "num_ucr_params_degree",
        "num_ucr_params_full",
        "best_restart",
        "seed_opt",
        "num_steps",
        "termination_reason",
        "wall_clock_sec",
        "optimizer_name",
        "learning_rate",
        "learning_rate_schedule",
        "max_steps",
        "eval_interval",
        "threshold",
        "num_restarts",
        "seed_start",
        "projection_strategy",
        "benchmark_seed",
        "data_seed",
        "raw_outcomes",
        "effective_m_outcomes",
        "coverage_ratio",
        "class_group_sizes",
        "prior_type",
        "fiducial_id",
        "orbit_index_set_id",
    ]


def _coerce_csv_value(key: str, value: str) -> Any:
    if value is None or value == "":
        return value
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    return value


def _load_rows_from_csvs(paths: Sequence[str]) -> list[dict[str, Any]]:
    return _read_typed_csv_rows(paths, _coerce_csv_value)


def _load_restart_rows_from_jsonls(paths: Sequence[str]) -> list[dict[str, Any]]:
    return _load_jsonl_rows(paths, skip_missing=True)


def _write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_dict_csv(path, rows, fieldnames=_fieldnames_with_extras(rows, _results_fieldnames()))


def _write_restart_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "se": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr) / math.sqrt(arr.size)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _iqr_payload(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def summarize_rows_by_degree(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["degree"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for degree in sorted(grouped):
        bucket = grouped[degree]
        first = bucket[0]
        payload = {
            "degree": int(degree),
            "model_type": str(first["model_type"]),
            "count": int(len(bucket)),
        }
        for metric in ("gap_abs", "gap_rel", "p_succ"):
            for stat_name, stat_value in _stats([float(row[metric]) for row in bucket]).items():
                payload[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(payload)
    return summary_rows


def summarize_rows_by_m_degree(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["M"]), int(row["degree"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (m_value, degree), bucket in sorted(grouped.items()):
        first = bucket[0]
        payload = {
            "M": int(m_value),
            "degree": int(degree),
            "model_type": str(first["model_type"]),
            "count": int(len(bucket)),
        }
        for metric in ("gap_abs", "gap_rel", "p_succ"):
            for stat_name, stat_value in _stats([float(row[metric]) for row in bucket]).items():
                payload[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(payload)
    return summary_rows


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    _write_dict_csv(path, rows, fieldnames=list(rows[0].keys()))


def _clip_gap_for_plot(value: float, *, numerical_floor: float) -> float:
    return max(float(value), float(numerical_floor))


def _add_numerical_floor_band(ax: Any, *, numerical_floor: float) -> None:
    floor = float(numerical_floor)
    ax.axhspan(floor / 1.4, floor, color="#d8d8d8", alpha=0.45, zorder=0)
    ax.set_ylim(bottom=floor / 1.4)


def _plot_gap_by_m_instance(
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    dpi: int,
    numerical_floor: float,
) -> None:
    if not rows:
        return
    m_values = sorted({int(row["M"]) for row in rows})
    ncols = 2
    nrows = int(math.ceil(len(m_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 4.2 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax in axes_arr.ravel()[len(m_values) :]:
        ax.axis("off")

    for ax, m_value in zip(axes_arr.ravel(), m_values):
        rows_m = [row for row in rows if int(row["M"]) == int(m_value)]
        instances = sorted({int(row["instance_id"]) for row in rows_m})
        degrees = sorted({int(row["degree"]) for row in rows_m})
        full_x = max(degrees) + 1
        for idx, instance_id in enumerate(instances):
            color = color_cycle[idx % len(color_cycle)]
            bucket = sorted(
                [row for row in rows_m if int(row["instance_id"]) == int(instance_id)],
                key=lambda row: int(row["degree"]),
            )
            x_values = [int(row["degree"]) for row in bucket] + [full_x]
            y_values = [_clip_gap_for_plot(float(row["gap_abs"]), numerical_floor=numerical_floor) for row in bucket]
            y_values.append(_clip_gap_for_plot(float(bucket[0]["gap_abs_full_ref"]), numerical_floor=numerical_floor))
            ax.plot(
                x_values,
                y_values,
                color=color,
                marker="o",
                linewidth=1.4,
                markersize=4.2,
                label=f"inst {instance_id}",
            )
        ax.set_title(f"M={m_value}")
        ax.set_xticks([*degrees, full_x])
        ax.set_xticklabels([str(degree) for degree in degrees] + ["full"])
        ax.set_yscale("log")
        _add_numerical_floor_band(ax, numerical_floor=numerical_floor)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlabel("Walsh degree")
        ax.set_ylabel(f"gap_abs clipped at {numerical_floor:.0e} (log)")

    axes_arr.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_mean_gap_by_m(
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    dpi: int,
    numerical_floor: float,
) -> None:
    if not rows:
        return
    summary = summarize_rows_by_m_degree(rows)
    m_values = sorted({int(row["M"]) for row in summary})
    degrees = sorted({int(row["degree"]) for row in rows})
    full_x = max(degrees) + 1
    fig, ax = plt.subplots(figsize=(6.6, 4.05))
    for m_value in m_values:
        bucket = sorted([row for row in summary if int(row["M"]) == m_value], key=lambda row: int(row["degree"]))
        full_refs_by_instance = {
            int(row["instance_id"]): float(row["gap_abs_full_ref"])
            for row in rows
            if int(row["M"]) == int(m_value)
        }
        full_stats = _stats(list(full_refs_by_instance.values()))
        y_values = [
            *[_clip_gap_for_plot(float(row["gap_abs_mean"]), numerical_floor=numerical_floor) for row in bucket],
            _clip_gap_for_plot(float(full_stats["mean"]), numerical_floor=numerical_floor),
        ]
        se_values = [*[float(row["gap_abs_se"]) for row in bucket], float(full_stats["se"])]
        yerr_lower = [
            min(se_value, max(y_value - float(numerical_floor), 0.0))
            for y_value, se_value in zip(y_values, se_values)
        ]
        ax.errorbar(
            [*[int(row["degree"]) for row in bucket], full_x],
            y_values,
            yerr=[yerr_lower, se_values],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"M={m_value}",
        )
    ax.set_xlabel("Walsh degree", fontsize=14)
    ax.set_ylabel(r"$\Delta_{\mathrm{opt}}$", fontsize=16)
    ax.set_xticks([*degrees, full_x])
    ax.set_xticklabels([str(degree) for degree in degrees] + ["full"], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_yscale("log")
    _add_numerical_floor_band(ax, numerical_floor=numerical_floor)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title=r"mean $\pm$ SE", title_fontsize=11, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _plot_median_iqr_gap_by_m(
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    dpi: int,
    numerical_floor: float,
) -> None:
    if not rows:
        return
    m_values = sorted({int(row["M"]) for row in rows})
    degrees = sorted({int(row["degree"]) for row in rows})
    full_x = max(degrees) + 1
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for m_value in m_values:
        x_values: list[int] = []
        y_values: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []

        for degree in degrees:
            values = [
                float(row["gap_abs"])
                for row in rows
                if int(row["M"]) == int(m_value) and int(row["degree"]) == int(degree)
            ]
            payload = _iqr_payload(values)
            median = _clip_gap_for_plot(payload["median"], numerical_floor=numerical_floor)
            q1 = _clip_gap_for_plot(payload["q1"], numerical_floor=numerical_floor)
            q3 = _clip_gap_for_plot(payload["q3"], numerical_floor=numerical_floor)
            x_values.append(int(degree))
            y_values.append(median)
            lower_errors.append(max(median - q1, 0.0))
            upper_errors.append(max(q3 - median, 0.0))

        full_values_by_instance = {
            int(row["instance_id"]): float(row["gap_abs_full_ref"])
            for row in rows
            if int(row["M"]) == int(m_value)
        }
        full_payload = _iqr_payload(list(full_values_by_instance.values()))
        full_median = _clip_gap_for_plot(full_payload["median"], numerical_floor=numerical_floor)
        full_q1 = _clip_gap_for_plot(full_payload["q1"], numerical_floor=numerical_floor)
        full_q3 = _clip_gap_for_plot(full_payload["q3"], numerical_floor=numerical_floor)
        x_values.append(full_x)
        y_values.append(full_median)
        lower_errors.append(max(full_median - full_q1, 0.0))
        upper_errors.append(max(full_q3 - full_median, 0.0))

        ax.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"M={m_value}",
        )

    ax.set_xlabel("Walsh degree", fontsize=13)
    ax.set_ylabel(r"$\Delta_{\mathrm{opt}}$", fontsize=15)
    ax.set_xticks([*degrees, full_x])
    ax.set_xticklabels([str(degree) for degree in degrees] + ["full"], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_yscale("log")
    _add_numerical_floor_band(ax, numerical_floor=numerical_floor)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def _materialize_outputs(
    *,
    rows: Sequence[dict[str, Any]],
    restart_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    dirs = _prepare_output_dirs(output_dir)
    rows_sorted = sorted(rows, key=lambda row: (int(row["M"]), int(row["instance_id"]), int(row["degree"])))
    restart_rows_sorted = sorted(
        restart_rows,
        key=lambda row: (
            int(row.get("M", 0)),
            int(row.get("instance_id", 0)),
            int(row.get("degree", 0)),
            int(row.get("seed_opt", row.get("restart_id", 0))),
        ),
    )
    by_m_degree = summarize_rows_by_m_degree(rows_sorted)

    raw_csv_path = dirs["raw"] / RESULTS_FILENAME
    restart_jsonl_path = dirs["raw"] / RESTARTS_FILENAME
    by_m_degree_csv_path = dirs["raw"] / BY_M_DEGREE_FILENAME
    by_instance_plot_path = dirs["figures"] / BY_INSTANCE_PLOT_FILENAME
    mean_plot_path = dirs["figures"] / MEAN_PLOT_FILENAME
    median_iqr_plot_path = dirs["figures"] / MEDIAN_IQR_PLOT_FILENAME
    summary_json_path = dirs["summaries"] / SUMMARY_FILENAME

    _write_results_csv(raw_csv_path, rows_sorted)
    _write_restart_jsonl(restart_jsonl_path, restart_rows_sorted)
    _write_summary_csv(by_m_degree_csv_path, by_m_degree)
    _plot_gap_by_m_instance(
        rows_sorted,
        by_instance_plot_path,
        dpi=int(args.plot_dpi),
        numerical_floor=float(args.numerical_floor),
    )
    _plot_mean_gap_by_m(
        rows_sorted,
        mean_plot_path,
        dpi=int(args.plot_dpi),
        numerical_floor=float(args.numerical_floor),
    )
    _plot_median_iqr_gap_by_m(
        rows_sorted,
        median_iqr_plot_path,
        dpi=int(args.plot_dpi),
        numerical_floor=float(args.numerical_floor),
    )

    summary = {
        "config": {
            "n_sys": int(args.n_sys),
            "m_values": [int(value) for value in args.m_values],
            "instance_ids": [int(value) for value in args.instance_ids],
            "degrees": [int(value) for value in args.degrees],
            "model_name": "walsh_k_local",
            "reference_results_csv": str(Path(args.reference_results_csv).expanduser().resolve()),
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
            "numerical_floor": float(args.numerical_floor),
            "aggregate_only": bool(args.aggregate_only),
        },
        "counts": {
            "num_rows": int(len(rows_sorted)),
            "num_restart_rows": int(len(restart_rows_sorted)),
        },
        "aggregated_by_degree": summarize_rows_by_degree(rows_sorted),
        "aggregated_by_m_degree": by_m_degree,
        "artifacts": {
            "output_dir": str(output_dir),
            "results_csv": str(raw_csv_path),
            "restart_records_jsonl": str(restart_jsonl_path),
            "by_m_degree_csv": str(by_m_degree_csv_path),
            "gap_by_instance_png": str(by_instance_plot_path),
            "mean_gap_png": str(mean_plot_path),
            "median_iqr_gap_png": str(median_iqr_plot_path),
            "summary_json": str(summary_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {raw_csv_path}")
    print(f"saved: {restart_jsonl_path}")
    print(f"saved: {by_m_degree_csv_path}")
    print(f"saved: {by_instance_plot_path}")
    print(f"saved: {mean_plot_path}")
    print(f"saved: {median_iqr_plot_path}")
    print(f"saved: {summary_json_path}")
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    args.projection_strategy = normalize_projection_strategy(args.projection_strategy)
    args.m_values = sorted({int(value) for value in args.m_values})
    args.instance_ids = sorted({int(value) for value in args.instance_ids})
    args.degrees = sorted({int(value) for value in args.degrees})
    if int(args.n_sys) < 1:
        raise ValueError("n_sys must be >= 1.")
    if not args.m_values:
        raise ValueError("At least one M value is required.")
    if any(value < 2 for value in args.m_values):
        raise ValueError(f"M values must be >= 2, got {args.m_values}.")
    if not args.instance_ids:
        raise ValueError("At least one instance id is required.")
    if any(value < 0 for value in args.instance_ids):
        raise ValueError(f"instance ids must be >= 0, got {args.instance_ids}.")
    if not args.degrees:
        raise ValueError("At least one Walsh degree is required.")
    if any(value < 1 or value > 5 for value in args.degrees):
        raise ValueError(f"This Walsh degree sweep supports degrees 1..5, got {args.degrees}.")
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
        raise ValueError("Walsh degree sweep fixes optimizer='adam'.")
    if str(args.trainer) != "full":
        raise ValueError("Walsh degree sweep supports trainer='full' only.")
    if bool(args.renormalize_projected_probs):
        raise ValueError("Walsh degree sweep requires renormalize_projected_probs=False.")
    if not bool(args.use_scrambler):
        raise ValueError("Walsh degree sweep requires use_scrambler=True.")
    if args.aggregate_only and not args.input_result_csvs:
        raise ValueError("--aggregate-only requires --input-result-csvs.")


def run_walsh_degree_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_rows = load_reference_rows(
        Path(args.reference_results_csv),
        n_sys=int(args.n_sys),
        m_values=args.m_values,
        instance_ids=args.instance_ids,
    )
    reference_by_key = {
        (int(row["M"]), int(row["instance_id"])): row
        for row in reference_rows
    }
    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []

    for m_value in args.m_values:
        for instance_id in args.instance_ids:
            reference_row = reference_by_key[(int(m_value), int(instance_id))]
            benchmark_seed, data_seed = _seed_pair_for_instance(
                n_sys=int(args.n_sys),
                M=int(m_value),
                instance_id=int(instance_id),
            )
            if int(reference_row["benchmark_seed"]) != benchmark_seed or int(reference_row["data_seed"]) != data_seed:
                raise ValueError(
                    "Reference seed mismatch for "
                    f"M={m_value} instance_id={instance_id}: expected ({benchmark_seed}, {data_seed}), "
                    f"got ({reference_row['benchmark_seed']}, {reference_row['data_seed']})."
                )

            problem_args = _problem_namespace(
                n_sys=int(args.n_sys),
                m_outcome=int(m_value),
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
            target_states = jnp.arange(int(m_value), dtype=jnp.int32)
            raw_outcomes = 2 ** int(problem["n_anc"])
            groups, mapping_payload = build_projection_groups(
                raw_outcomes,
                int(m_value),
                strategy=str(args.projection_strategy),
            )

            for degree in args.degrees:
                spec = walsh_degree_model_spec(int(degree))
                model_args = argparse.Namespace(**vars(args))
                model_args.n_sys = int(args.n_sys)
                model_args.m_outcome = int(m_value)
                checkpoint_path = build_restart_checkpoint_path(
                    output_dir,
                    n_sys=int(args.n_sys),
                    M=int(m_value),
                    instance_id=int(instance_id),
                    model_type=str(spec.model_type),
                )
                summary, model_restart_rows = _run_model_restarts_restart_reuse(
                    spec=spec,
                    problem=problem,
                    args=model_args,
                    groups=groups,
                    target_states=target_states,
                    checkpoint_path=checkpoint_path,
                    instance_id=int(instance_id),
                    benchmark_seed=int(benchmark_seed),
                    data_seed=int(data_seed),
                )
                degree_row = _trained_degree_row(
                    reference_row=reference_row,
                    degree=int(degree),
                    summary=summary,
                    args=args,
                    raw_outcomes=int(raw_outcomes),
                    mapping_payload=mapping_payload,
                )
                rows.append(degree_row)
                for restart_row in model_restart_rows:
                    enriched = dict(restart_row)
                    enriched.update(
                        {
                            "instance_id": int(instance_id),
                            "n_sys": int(args.n_sys),
                            "d": int(2 ** int(args.n_sys)),
                            "M": int(m_value),
                            "M_over_d": float(int(m_value) / float(2 ** int(args.n_sys))),
                            "degree": int(degree),
                            "benchmark_seed": int(benchmark_seed),
                            "data_seed": int(data_seed),
                        }
                    )
                    restart_rows.append(enriched)
                print(
                    f"[walsh_degree_sweep] n_sys={args.n_sys} M={m_value} "
                    f"instance_id={instance_id} degree={degree} "
                    f"gap_abs={degree_row['gap_abs']:.6f}",
                    flush=True,
                )

    return _materialize_outputs(rows=rows, restart_rows=restart_rows, args=args, output_dir=output_dir)


def aggregate_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    rows = _load_rows_from_csvs(args.input_result_csvs)
    restart_rows = _load_restart_rows_from_jsonls(args.input_restart_jsonls)
    output_dir = Path(args.output_dir).expanduser().resolve()
    return _materialize_outputs(rows=rows, restart_rows=restart_rows, args=args, output_dir=output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WH Walsh-basis cumulative degree sweep.")
    parser.add_argument("--n-sys", type=int, default=DEFAULT_N_SYS)
    parser.add_argument("--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    parser.add_argument("--instance-ids", type=int, nargs="+", default=list(DEFAULT_INSTANCE_IDS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--reference-results-csv", type=str, default=str(DEFAULT_REFERENCE_RESULTS_CSV))
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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--projection-strategy", type=str, default="drop_extra")
    parser.add_argument("--renormalize-projected-probs", type=_parse_bool_arg, default=False, metavar="{True,False}")
    parser.add_argument("--state-dtype", type=str, choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--use-scrambler", type=_parse_bool_arg, default=True, metavar="{True,False}")
    parser.add_argument("--plot-dpi", type=int, default=DEFAULT_PLOT_DPI)
    parser.add_argument("--numerical-floor", type=float, default=DEFAULT_NUMERICAL_FLOOR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--input-result-csvs", type=str, nargs="+", default=None)
    parser.add_argument("--input-restart-jsonls", type=str, nargs="*", default=())
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(
            default_output_root(
                n_sys=int(args.n_sys),
                m_values=args.m_values,
                projection_strategy=str(args.projection_strategy),
                instance_ids=args.instance_ids,
                num_restarts=int(args.num_restarts),
            )
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.aggregate_only):
        aggregate_existing_outputs(args)
        return
    run_walsh_degree_sweep(args)


if __name__ == "__main__":
    main()
