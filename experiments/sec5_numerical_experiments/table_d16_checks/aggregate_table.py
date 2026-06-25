from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = (
    ROOT / "experiments" / "sec5_numerical_experiments" / "table_d16_checks" / "results"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "paper" / "table_d16_checks"
INSTANCE_BEST_COLUMNS = [
    "state_family",
    "n_sys",
    "d",
    "M",
    "M_over_d",
    "regime",
    "instance_id",
    "model_type",
    "model_label",
    "walsh_degree",
    "restart_id",
    "seed_opt",
    "num_steps",
    "termination_reason",
    "final_objective_value",
    "best_objective_value",
    "p_succ",
    "p_opt_sdp",
    "gap_abs_sdp",
    "num_restarts",
    "observed_restart_count",
    "theta_dim",
    "num_ops",
    "optimizer_name",
    "learning_rate",
    "learning_rate_schedule",
    "max_steps",
    "eval_interval",
    "threshold",
    "su_depth",
    "scale_init",
    "bias_scale_init",
    "memory_optimization",
    "dtype",
    "theta_dtype",
    "jax_backend",
    "jax_devices",
    "benchmark_seed",
    "data_seed",
    "state_seed",
    "state_array_sha256",
    "_source_csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_label(model_type: str) -> str:
    labels = {
        "full_ucr": "full-UCR",
        "walsh_degree_1": "WD-1",
        "walsh_degree_4": "WD-4",
        "walsh_degree_5": "WD-5",
    }
    return labels.get(str(model_type), str(model_type))


def _regime_label(M: int, d: int) -> str:
    if int(M) == int(d):
        return "M=d"
    if int(M) == int(d) + 1:
        return "M=d+1"
    return f"M/d={float(M) / float(d):.6g}"


def _stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "median": "",
            "q1": "",
            "q3": "",
            "min": "",
            "max": "",
        }
    return {
        "count": int(arr.size),
        "median": float(np.quantile(arr, 0.5, method="linear")),
        "q1": float(np.quantile(arr, 0.25, method="linear")),
        "q3": float(np.quantile(arr, 0.75, method="linear")),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "state_family",
        "M",
        "regime",
        "model_type",
        "model_label",
        "instances",
        "restarts_per_instance",
        "gap_abs_sdp_median",
        "gap_abs_sdp_q1",
        "gap_abs_sdp_q3",
        "p_succ_median",
        "p_succ_q1",
        "p_succ_q3",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in ordered} for row in rows])


def _infer_state_family(path: Path, row: dict[str, Any]) -> str:
    explicit = str(row.get("state_family", "")).strip()
    if explicit:
        return explicit
    path_text = str(path).lower()
    if "haar" in path_text:
        return "haar"
    if "weyl" in path_text or "wh_md" in path.name.lower():
        return "weyl_heisenberg"
    return ""


def _display_state_family(value: str) -> str:
    labels = {
        "haar": "Haar-random",
        "exact_haar": "Haar-random",
        "weyl": "Weyl--Heisenberg",
        "weyl_heisenberg": "Weyl--Heisenberg",
        "wh": "Weyl--Heisenberg",
    }
    return labels.get(str(value), str(value))


def _compact_instance_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {key: row.get(key, "") for key in INSTANCE_BEST_COLUMNS}
    compact["model_label"] = _model_label(str(row.get("model_type", "")))
    compact["regime"] = _regime_label(int(row["M"]), int(row["d"]))
    compact["state_family_label"] = _display_state_family(str(row.get("state_family", "")))
    return compact


def _results_csv_paths(results_roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in results_roots:
        if root.is_file():
            paths.append(root)
            continue
        paths.extend(sorted(root.glob("**/raw/*_results.csv")))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def load_restart_rows(results_roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _results_csv_paths(results_roots):
        for row in _read_csv(path):
            row = dict(row)
            row["_source_csv"] = str(path)
            row["state_family"] = _infer_state_family(path, row)
            rows.append(row)
    return rows


def select_instance_best(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    observed_restart_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            row.get("state_family", ""),
            _int_or_none(row.get("n_sys")),
            _int_or_none(row.get("M")),
            _int_or_none(row.get("instance_id")),
            row.get("model_type", ""),
        )
        if any(value is None for value in key[1:4]) or key[-1] == "":
            continue
        observed_restart_counts[key] = observed_restart_counts.get(key, 0) + 1
        current = best.get(key)
        objective = _float_or_nan(row.get("final_objective_value"))
        current_objective = _float_or_nan(current.get("final_objective_value")) if current else float("inf")
        if current is None or objective < current_objective:
            best[key] = row
    selected: list[dict[str, Any]] = []
    for key in sorted(best):
        row = dict(best[key])
        row["observed_restart_count"] = observed_restart_counts.get(key, "")
        selected.append(row)
    return selected


def aggregate(instance_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in instance_rows:
        key = (str(row.get("state_family", "")), int(row["M"]), str(row.get("model_type", "")))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        state_family, M, model_type = key
        bucket = grouped[key]
        first = bucket[0]
        d = int(first["d"])
        gap_stats = _stats(_float_or_nan(row.get("gap_abs_sdp")) for row in bucket)
        psucc_stats = _stats(_float_or_nan(row.get("p_succ")) for row in bucket)
        restarts = sorted(
            {
                _int_or_none(row.get("observed_restart_count")) or _int_or_none(row.get("num_restarts"))
                for row in bucket
                if (_int_or_none(row.get("observed_restart_count")) or _int_or_none(row.get("num_restarts")))
            }
        )
        summary_rows.append(
            {
                "state_family": state_family,
                "state_family_label": _display_state_family(state_family),
                "n_sys": int(first["n_sys"]),
                "d": d,
                "M": M,
                "M_over_d": float(M / d),
                "regime": _regime_label(M, d),
                "model_type": model_type,
                "model_label": _model_label(model_type),
                "instances": int(len(bucket)),
                "restarts_per_instance": ";".join(str(value) for value in restarts),
                "gap_abs_sdp_count": gap_stats["count"],
                "gap_abs_sdp_median": gap_stats["median"],
                "gap_abs_sdp_q1": gap_stats["q1"],
                "gap_abs_sdp_q3": gap_stats["q3"],
                "gap_abs_sdp_min": gap_stats["min"],
                "gap_abs_sdp_max": gap_stats["max"],
                "p_succ_count": psucc_stats["count"],
                "p_succ_median": psucc_stats["median"],
                "p_succ_q1": psucc_stats["q1"],
                "p_succ_q3": psucc_stats["q3"],
                "p_succ_min": psucc_stats["min"],
                "p_succ_max": psucc_stats["max"],
            }
        )
    return summary_rows


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6e}"


def write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# d=16 Table Summary",
        "",
        "| Ensemble | M | Regime | Model | Instances | Restarts | Delta_opt median [Q1,Q3] | P_succ median [Q1,Q3] |",
        "|---|---:|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {state_family} | {M} | {regime} | {model_label} | {instances} | {restarts} | "
            "{gap_med} [{gap_q1}, {gap_q3}] | {ps_med} [{ps_q1}, {ps_q3}] |".format(
                state_family=row.get("state_family_label", row["state_family"]),
                M=row["M"],
                regime=row["regime"],
                model_label=row["model_label"],
                instances=row["instances"],
                restarts=row["restarts_per_instance"],
                gap_med=_fmt(row["gap_abs_sdp_median"]),
                gap_q1=_fmt(row["gap_abs_sdp_q1"]),
                gap_q3=_fmt(row["gap_abs_sdp_q3"]),
                ps_med=_fmt(row["p_succ_median"]),
                ps_q1=_fmt(row["p_succ_q1"]),
                ps_q3=_fmt(row["p_succ_q3"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate d=16 GPU result roots into table-ready CSV/Markdown.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--input-roots",
        type=Path,
        nargs="+",
        help="Explicit result roots or *_results.csv files to aggregate. Overrides --results-root.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if args.input_roots:
        results_roots = [path.expanduser().resolve() for path in args.input_roots]
    else:
        results_roots = [args.results_root.expanduser().resolve()]
    output_dir = args.output_dir.expanduser().resolve()
    restart_rows = load_restart_rows(results_roots)
    instance_rows = select_instance_best(restart_rows)
    compact_instance_rows = [_compact_instance_row(row) for row in instance_rows]
    summary_rows = aggregate(instance_rows)

    _write_csv(output_dir / "table_d16_checks_instance_best.csv", compact_instance_rows)
    _write_csv(output_dir / "table_d16_checks_summary.csv", summary_rows)
    write_markdown(output_dir / "table_d16_checks_summary.md", summary_rows)

    print("results_roots:")
    for root in results_roots:
        print(f"  {root}")
    print(f"loaded_restart_rows={len(restart_rows)}")
    print(f"instance_best_rows={len(instance_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"saved: {output_dir / 'table_d16_checks_instance_best.csv'}")
    print(f"saved: {output_dir / 'table_d16_checks_summary.csv'}")
    print(f"saved: {output_dir / 'table_d16_checks_summary.md'}")


if __name__ == "__main__":
    main()
