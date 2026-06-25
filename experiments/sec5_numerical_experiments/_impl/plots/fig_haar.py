from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
DEFAULT_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_haar_d8_sweep"
    / "raw"
    / "exact_haar_d8_sweep_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_CSV.parent.parent / "figures"
DEFAULT_OUTPUT_STEM = "exact_haar_d8_gap_zoom_two_panel"
DEFAULT_SINGLE_OUTPUT_STEM = "exact_haar_d8_gap_single_panel"
DEFAULT_DISPLAY_FLOOR = 1e-7
DEFAULT_SINGLE_YMAX = 0.3
CONNECTOR_COLOR = "0.45"
MODEL_CONFIGS = {
    "full_ucr": {
        "column": "gap_abs_full",
        "label": "full-UCR",
        "color": "tab:blue",
        "marker": "o",
        "x_offset": -0.22,
    },
    "random_sparse_ucr": {
        "column": "gap_abs_random_sparse",
        "label": "RS-UCR",
        "color": "tab:red",
        "marker": "v",
        "x_offset": 0.0,
    },
    "walsh_degree_1": {
        "column": "gap_abs_walsh_deg1",
        "label": "WD-1",
        "color": "tab:orange",
        "marker": "D",
        "x_offset": 0.22,
    },
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, Any], key: str) -> int:
    return int(row[key])


def _stats(values: Sequence[float]) -> dict[str, float]:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return {key: float("nan") for key in ("mean", "se", "median", "q1", "q3", "min", "max")}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((value - mean) ** 2 for value in vals) / n

    def quantile(q: float) -> float:
        if n == 1:
            return vals[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": mean,
        "se": math.sqrt(var) / math.sqrt(n),
        "median": quantile(0.5),
        "q1": quantile(0.25),
        "q3": quantile(0.75),
        "min": vals[0],
        "max": vals[-1],
    }


def _aggregate_by_m(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_int(row, "M"), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for M in sorted(grouped):
        bucket = grouped[M]
        payload: dict[str, Any] = {
            "M": int(M),
            "d": _int(bucket[0], "d"),
            "M_over_d": _float(bucket[0], "M_over_d"),
            "count": len(bucket),
        }
        for model_key, config in MODEL_CONFIGS.items():
            column = str(config["column"])
            payload[model_key] = _stats([_float(row, column) for row in bucket])
        aggregated.append(payload)
    return aggregated


def _with_line_gap(
    x_values: Sequence[float],
    y_values: Sequence[float],
    m_values: Sequence[int],
) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    prev_m: int | None = None
    for x_value, y_value, m_value in zip(x_values, y_values, m_values, strict=True):
        if prev_m is not None and int(m_value) - int(prev_m) > 1:
            out_x.append(float("nan"))
            out_y.append(float("nan"))
        out_x.append(float(x_value))
        out_y.append(float(y_value))
        prev_m = int(m_value)
    return out_x, out_y


def _with_transition_gap(
    x_values: Sequence[float],
    y_values: Sequence[float],
    m_values: Sequence[int],
    *,
    break_after_m: int,
    break_before_m: int,
) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    prev_m: int | None = None
    for x_value, y_value, m_value in zip(x_values, y_values, m_values, strict=True):
        if prev_m == int(break_after_m) and int(m_value) == int(break_before_m):
            out_x.append(float("nan"))
            out_y.append(float("nan"))
        out_x.append(float(x_value))
        out_y.append(float(y_value))
        prev_m = int(m_value)
    return out_x, out_y


def _instance_offsets(rows: Sequence[dict[str, str]]) -> dict[int, float]:
    instance_ids = sorted({_int(row, "instance_id") for row in rows})
    if len(instance_ids) <= 1:
        return {instance_ids[0]: 0.0} if instance_ids else {}
    offsets = [(-0.045 + 0.09 * idx / (len(instance_ids) - 1)) for idx in range(len(instance_ids))]
    return {instance_id: offsets[idx] for idx, instance_id in enumerate(instance_ids)}


def _displayed_gap(value: float, floor: float | None) -> float:
    if floor is None:
        return float(value)
    return max(float(value), float(floor))


def _style_axes(ax: plt.Axes, *, m_values: Sequence[int]) -> None:
    ax.set_xlabel("M")
    ax.set_ylabel(r"$\Delta_{\mathrm{opt}}$")
    ax.set_xticks(list(m_values))
    ax.set_xlim(min(m_values) - 0.55, max(m_values) + 0.55)
    ax.axhline(0.0, color="0.25", linewidth=0.6, linestyle=":", zorder=1)
    ax.grid(True, which="major", axis="both", color="0.85", linewidth=0.55)
    ax.set_axisbelow(True)


def _plot_panel(
    ax: plt.Axes,
    *,
    rows: Sequence[dict[str, str]],
    aggregated_rows: Sequence[dict[str, Any]],
    show_instances: bool,
    display_floor: float | None,
) -> None:
    instance_offset_by_id = _instance_offsets(rows)
    for model_key, config in MODEL_CONFIGS.items():
        color = str(config["color"])
        marker = str(config["marker"])
        column = str(config["column"])
        model_offset = float(config["x_offset"])

        if show_instances:
            ax.scatter(
                [
                    _int(row, "M") + model_offset + instance_offset_by_id[_int(row, "instance_id")]
                    for row in rows
                ],
                [_displayed_gap(_float(row, column), display_floor) for row in rows],
                s=10,
                color=color,
                marker=marker,
                alpha=0.40,
                linewidths=0.0,
                zorder=2,
            )

        median_rows = [row for row in aggregated_rows if model_key in row]
        median_m = [int(row["M"]) for row in median_rows]
        median_x = [int(row["M"]) + model_offset for row in median_rows]
        median_y = [
            _displayed_gap(float(row[model_key]["median"]), display_floor)
            for row in median_rows
        ]
        median_x_with_gap, median_y_with_gap = _with_line_gap(median_x, median_y, median_m)
        ax.plot(
            median_x_with_gap,
            median_y_with_gap,
            color=color,
            marker=marker,
            markersize=3.4,
            linewidth=1.25,
            label=str(config["label"]),
            zorder=3,
        )


def _plot_single_linear_panel(
    ax: plt.Axes,
    *,
    rows: Sequence[dict[str, str]],
    aggregated_rows: Sequence[dict[str, Any]],
    break_after_m: int,
    break_before_m: int,
) -> None:
    for model_key, config in MODEL_CONFIGS.items():
        color = str(config["color"])
        marker = str(config["marker"])
        model_offset = float(config["x_offset"])

        median_rows = [row for row in aggregated_rows if model_key in row]
        median_m = [int(row["M"]) for row in median_rows]
        median_x = [int(row["M"]) + model_offset for row in median_rows]
        median_y = [float(row[model_key]["median"]) for row in median_rows]
        q1_y = [float(row[model_key]["q1"]) for row in median_rows]
        q3_y = [float(row[model_key]["q3"]) for row in median_rows]

        segment_start = 0
        for idx in range(1, len(median_rows) + 1):
            is_break = (
                idx < len(median_rows)
                and median_m[idx - 1] == int(break_after_m)
                and median_m[idx] == int(break_before_m)
            )
            is_end = idx == len(median_rows)
            if not (is_break or is_end):
                continue
            ax.fill_between(
                median_x[segment_start:idx],
                q1_y[segment_start:idx],
                q3_y[segment_start:idx],
                color=color,
                alpha=0.14,
                linewidth=0.0,
                zorder=2,
            )
            segment_start = idx

        median_x_with_gap, median_y_with_gap = _with_transition_gap(
            median_x,
            median_y,
            median_m,
            break_after_m=int(break_after_m),
            break_before_m=int(break_before_m),
        )
        ax.plot(
            median_x_with_gap,
            median_y_with_gap,
            color=color,
            marker=marker,
            markersize=3.7,
            linewidth=1.35,
            label=str(config["label"]),
            zorder=3,
        )


def _dynamic_log_ylim(rows: Sequence[dict[str, str]], *, floor: float) -> tuple[float, float]:
    positive_values = [
        max(_float(row, str(config["column"])), float(floor))
        for row in rows
        for config in MODEL_CONFIGS.values()
    ]
    upper = max(positive_values) * 1.8 if positive_values else floor * 10.0
    upper = max(upper, floor * 10.0)
    return float(floor), float(upper)


def _dynamic_linear_ylim(rows: Sequence[dict[str, str]]) -> tuple[float, float]:
    values = [_float(row, str(config["column"])) for row in rows for config in MODEL_CONFIGS.values()]
    if not values:
        return -0.01, 0.1
    lower = min(0.0, min(values))
    upper = max(values)
    span = max(upper - lower, 1e-3)
    return lower - 0.08 * span, upper + 0.16 * span


def make_two_panel(
    *,
    results_csv: Path,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    display_floor: float,
) -> dict[str, Any]:
    rows = _read_rows(results_csv)
    if not rows:
        raise ValueError(f"No rows found in results CSV: {results_csv}")

    missing_columns = [
        str(config["column"])
        for config in MODEL_CONFIGS.values()
        if str(config["column"]) not in rows[0]
    ]
    if missing_columns:
        raise ValueError(f"Results CSV is missing required columns: {missing_columns}")

    aggregated_rows = _aggregate_by_m(rows)
    m_values = sorted({_int(row, "M") for row in rows})
    detail_ylim = _dynamic_log_ylim(rows, floor=float(display_floor))
    full_ylim = _dynamic_linear_ylim(rows)
    zoom_ylim = (full_ylim[0], min(full_ylim[1], detail_ylim[1]))

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 7.2,
            "lines.linewidth": 1.1,
        }
    )
    fig, (ax_detail, ax_full) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.55),
        constrained_layout=True,
    )

    _plot_panel(
        ax_detail,
        rows=rows,
        aggregated_rows=aggregated_rows,
        show_instances=True,
        display_floor=float(display_floor),
    )
    _style_axes(ax_detail, m_values=m_values)
    ax_detail.set_yscale("log")
    ax_detail.set_ylim(*detail_ylim)
    ax_detail.grid(True, which="minor", axis="y", color="0.92", linewidth=0.35)

    _plot_panel(
        ax_full,
        rows=rows,
        aggregated_rows=aggregated_rows,
        show_instances=False,
        display_floor=None,
    )
    _style_axes(ax_full, m_values=m_values)
    ax_full.set_ylabel("")
    ax_full.set_ylim(*full_ylim)
    ax_full.legend(
        frameon=False,
        handlelength=1.25,
        columnspacing=0.65,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        borderaxespad=0.0,
    )

    zoom_xlim = (min(m_values) - 0.45, max(m_values) + 0.45)
    zoom_box = Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        fill=False,
        edgecolor=CONNECTOR_COLOR,
        linewidth=0.85,
        linestyle=(0, (3, 2)),
        zorder=5,
    )
    ax_full.add_patch(zoom_box)

    for y_value, detail_y in ((zoom_ylim[1], 1.0), (zoom_ylim[0], 0.0)):
        connector = ConnectionPatch(
            xyA=(zoom_xlim[0], y_value),
            coordsA=ax_full.transData,
            xyB=(1.0, detail_y),
            coordsB=ax_detail.transAxes,
            color=CONNECTOR_COLOR,
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=4,
            clip_on=False,
        )
        fig.add_artist(connector)

    ax_detail.spines["right"].set_color(CONNECTOR_COLOR)
    ax_full.spines["left"].set_color(CONNECTOR_COLOR)
    ax_detail.spines["right"].set_linewidth(0.9)
    ax_full.spines["left"].set_linewidth(0.9)
    fig.align_ylabels([ax_detail])

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    summary_path = output_dir / f"{output_stem}_summary.json"
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "input_csvs": {"exact_haar_d8": str(results_csv)},
        "row_count": len(rows),
        "m_values": m_values,
        "instance_ids": sorted({_int(row, "instance_id") for row in rows}),
        "models": {
            model_key: {
                "label": str(config["label"]),
                "column": str(config["column"]),
            }
            for model_key, config in MODEL_CONFIGS.items()
        },
        "aggregated_by_m": aggregated_rows,
        "artifacts": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "summary_json": str(summary_path),
        "panel_a": {
            "description": "Log detail view with individual instances and median trends.",
            "yscale": "log",
            "ylim": list(detail_ylim),
            "display_floor": float(display_floor),
        },
        "panel_b": {
            "description": "Full-scale median plot.",
            "ylim": list(full_ylim),
            "zoom_box": {"xlim": list(zoom_xlim), "ylim": list(zoom_ylim)},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def make_single_panel(
    *,
    results_csv: Path,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    break_after_m: int = 8,
    break_before_m: int = 9,
    y_max: float = DEFAULT_SINGLE_YMAX,
) -> dict[str, Any]:
    rows = _read_rows(results_csv)
    if not rows:
        raise ValueError(f"No rows found in results CSV: {results_csv}")

    missing_columns = [
        str(config["column"])
        for config in MODEL_CONFIGS.values()
        if str(config["column"]) not in rows[0]
    ]
    if missing_columns:
        raise ValueError(f"Results CSV is missing required columns: {missing_columns}")

    aggregated_rows = _aggregate_by_m(rows)
    m_values = sorted({_int(row, "M") for row in rows})
    linear_ylim_raw = _dynamic_linear_ylim(rows)
    linear_ylim = (linear_ylim_raw[0], float(y_max))

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 6.3,
            "lines.linewidth": 1.1,
        }
    )
    fig, ax = plt.subplots(figsize=(3.85, 2.65), constrained_layout=True)
    _plot_single_linear_panel(
        ax,
        rows=rows,
        aggregated_rows=aggregated_rows,
        break_after_m=int(break_after_m),
        break_before_m=int(break_before_m),
    )
    _style_axes(ax, m_values=m_values)
    ax.set_ylim(*linear_ylim)
    ax.legend(
        title="median (IQR)",
        title_fontsize=7.6,
        fontsize=7.2,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        edgecolor="0.75",
        facecolor="white",
        handlelength=1.25,
        columnspacing=0.65,
        loc="upper right",
        ncol=3,
        borderaxespad=0.35,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    summary_path = output_dir / f"{output_stem}_summary.json"
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "input_csvs": {"exact_haar_d8": str(results_csv)},
        "row_count": len(rows),
        "m_values": m_values,
        "instance_ids": sorted({_int(row, "instance_id") for row in rows}),
        "models": {
            model_key: {
                "label": str(config["label"]),
                "column": str(config["column"]),
            }
            for model_key, config in MODEL_CONFIGS.items()
        },
        "aggregated_by_m": aggregated_rows,
        "artifacts": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "summary_json": str(summary_path),
        "panel": {
            "description": "Linear-scale single-panel gap plot with median trends and IQR bands.",
            "yscale": "linear",
            "ylim": list(linear_ylim),
            "raw_data_ylim": list(linear_ylim_raw),
            "interval": "IQR (q1-q3)",
            "legend_title": "median (IQR)",
            "legend_location": "upper right inside axes",
            "median_line_break": {
                "break_after_m": int(break_after_m),
                "break_before_m": int(break_before_m),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the two-panel exact Haar D8 gap figure."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--single-output-stem", type=str, default=DEFAULT_SINGLE_OUTPUT_STEM)
    parser.add_argument("--single-y-max", type=float, default=DEFAULT_SINGLE_YMAX)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--display-floor", type=float, default=DEFAULT_DISPLAY_FLOOR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = make_two_panel(
        results_csv=args.results_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        output_stem=str(args.output_stem),
        dpi=int(args.dpi),
        display_floor=float(args.display_floor),
    )
    single_summary = make_single_panel(
        results_csv=args.results_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        output_stem=str(args.single_output_stem),
        dpi=int(args.dpi),
        y_max=float(args.single_y_max),
    )
    print(f"saved: {summary['artifacts']['png']}")
    print(f"saved: {summary['artifacts']['pdf']}")
    print(f"saved: {summary['summary_json']}")
    print(f"saved: {single_summary['artifacts']['png']}")
    print(f"saved: {single_summary['artifacts']['pdf']}")
    print(f"saved: {single_summary['summary_json']}")


if __name__ == "__main__":
    main()
