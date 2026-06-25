from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle

IMPL_DIR = Path(__file__).resolve().parents[1]
if str(IMPL_DIR) not in sys.path:
    sys.path.append(str(IMPL_DIR))

from common import (
    DEFAULT_RESULTS_CSV,
    _aggregate_by_m,
    _all_rows,
    _float,
    _int,
    _read_rows,
    _with_line_gap,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
DEFAULT_DEGREE1_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_walsh_degree1_results.csv"
)
DEFAULT_RANDOM_SPARSE_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "walsh_random_sparse_vs_degree1_results.csv"
)
DEFAULT_UCR_RANDOM_SPARSE_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "random_sparse_vs_degree1_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DEGREE1_RESULTS_CSV.parent.parent / "figures"
DEFAULT_OUTPUT_STEM = "wh_md_sweep_gap_zoom_two_panel"
DETAIL_YLIM = (-0.0025, 0.05)
DETAIL_LOG_YLIM = (1e-7, 7e-2)
ZOOM_BOX_XLIM = (4.45, 12.55)
CONNECTOR_COLOR = "0.45"
ZERO_CLIPPED_FLAG_PREFIX = "__zero_clipped__"
MODEL_CONFIGS = {
    "full_ucr": {
        "source": "full",
        "column": "gap_abs_full",
        "label": "full-UCR",
        "color": "tab:blue",
        "marker": "o",
        "x_offset": -0.27,
    },
    "walsh_degree_1": {
        "source": "degree1",
        "column": "gap_abs_walsh_deg1",
        "label": "Walsh degree-1",
        "color": "tab:purple",
        "marker": "D",
        "x_offset": -0.09,
    },
    "walsh_random_sparse": {
        "source": "random_sparse",
        "column": "gap_abs_walsh_random_sparse",
        "label": "Walsh random sparse",
        "color": "tab:green",
        "marker": "^",
        "x_offset": 0.09,
    },
    "ucr_random_sparse": {
        "source": "ucr_random_sparse",
        "column": "gap_abs_random_sparse",
        "label": "full-UCR random sparse",
        "color": "tab:orange",
        "marker": "v",
        "x_offset": 0.27,
    },
}


def _apply_display_overrides(
    configs: dict[str, dict[str, Any]],
    *,
    baseline_color: str | None,
    wd1_label: str | None,
    wd1_color: str | None,
    ucr_random_sparse_label: str | None,
    ucr_random_sparse_color: str | None,
) -> None:
    if baseline_color is not None:
        configs["full_ucr"]["color"] = baseline_color
    if wd1_label is not None:
        configs["walsh_degree_1"]["label"] = wd1_label
    if wd1_color is not None:
        configs["walsh_degree_1"]["color"] = wd1_color
    if ucr_random_sparse_label is not None:
        configs["ucr_random_sparse"]["label"] = ucr_random_sparse_label
    if ucr_random_sparse_color is not None:
        configs["ucr_random_sparse"]["color"] = ucr_random_sparse_color


def _model_configs(
    model_set: str,
    *,
    omit_walsh_random_sparse: bool = False,
    baseline_color: str | None = None,
    wd1_label: str | None = None,
    wd1_color: str | None = None,
    ucr_random_sparse_label: str | None = None,
    ucr_random_sparse_color: str | None = None,
) -> dict[str, dict[str, Any]]:
    configs = {key: dict(value) for key, value in MODEL_CONFIGS.items()}
    _apply_display_overrides(
        configs,
        baseline_color=baseline_color,
        wd1_label=wd1_label,
        wd1_color=wd1_color,
        ucr_random_sparse_label=ucr_random_sparse_label,
        ucr_random_sparse_color=ucr_random_sparse_color,
    )
    if model_set == "all":
        if omit_walsh_random_sparse:
            selected = {key: configs[key] for key in ("full_ucr", "walsh_degree_1", "ucr_random_sparse")}
            selected["full_ucr"]["x_offset"] = -0.22
            selected["walsh_degree_1"]["x_offset"] = 0.0
            selected["ucr_random_sparse"]["x_offset"] = 0.22
            return selected
        return configs
    if model_set == "detail":
        if omit_walsh_random_sparse:
            selected = {key: configs[key] for key in ("full_ucr", "walsh_degree_1")}
            selected["full_ucr"]["x_offset"] = -0.11
            selected["walsh_degree_1"]["x_offset"] = 0.11
            return selected
        selected = {key: configs[key] for key in ("full_ucr", "walsh_degree_1", "walsh_random_sparse")}
        selected["full_ucr"]["x_offset"] = -0.22
        selected["walsh_degree_1"]["x_offset"] = 0.0
        selected["walsh_random_sparse"]["x_offset"] = 0.22
        return selected
    raise ValueError(f"Unknown model set: {model_set}")


def _series_rows(
    *,
    source_rows: dict[str, Sequence[dict[str, str]]],
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, Sequence[dict[str, str]]]:
    return {
        model_key: source_rows.get(str(config["source"]), [])
        for model_key, config in model_configs.items()
    }


def _instance_offsets(rows: Sequence[dict[str, str]]) -> dict[int, float]:
    offsets = [-0.038, -0.019, 0.0, 0.019, 0.038]
    return {
        instance_id: offsets[idx % len(offsets)]
        for idx, instance_id in enumerate(sorted({_int(row, "instance_id") for row in rows}))
    }


def _style_axes(ax: plt.Axes, *, m_values: Sequence[int]) -> None:
    ax.set_xlabel("M")
    ax.set_ylabel(r"$\Delta_{\mathrm{opt}}$")
    ax.set_xticks(list(m_values))
    ax.set_xlim(min(m_values) - 0.55, max(m_values) + 0.55)
    ax.axhline(0.0, color="0.25", linewidth=0.6, linestyle=":", zorder=1)
    ax.grid(True, which="major", axis="both", color="0.85", linewidth=0.55)
    ax.set_axisbelow(True)


def _add_regime_span(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    label: str,
    y: float,
) -> None:
    transform = ax.get_xaxis_transform()
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords=transform,
        textcoords=transform,
        arrowprops={
            "arrowstyle": "|-|",
            "color": "0.28",
            "linewidth": 0.75,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        annotation_clip=False,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y + 0.045,
        label,
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.2",
        clip_on=False,
    )


def _plot_panel(
    ax: plt.Axes,
    *,
    series_rows: dict[str, Sequence[dict[str, str]]],
    aggregated_rows: Sequence[dict[str, Any]],
    model_configs: dict[str, dict[str, Any]],
    show_instances: bool,
    instance_model_keys: set[str] | None = None,
    log_floor: float | None = None,
    mark_zero_clipped: bool = False,
) -> None:
    all_rows = _all_rows(series_rows)
    instance_offset_by_id = _instance_offsets(all_rows)
    instance_model_keys = set() if instance_model_keys is None else set(instance_model_keys)

    for model_key, config in model_configs.items():
        rows = series_rows.get(model_key, [])
        if not rows:
            continue

        color = str(config["color"])
        marker = str(config["marker"])
        x_offset = float(config["x_offset"])
        column = str(config["column"])

        if show_instances or model_key in instance_model_keys:
            zero_flag = f"{ZERO_CLIPPED_FLAG_PREFIX}{column}"
            for clipped_group, row_group in (
                (False, [row for row in rows if row.get(zero_flag) != "1"]),
                (True, [row for row in rows if row.get(zero_flag) == "1"]),
            ):
                if not row_group:
                    continue
                scatter_y_raw = [_float(row, column) for row in row_group]
                scatter_y = (
                    [max(value, float(log_floor)) for value in scatter_y_raw]
                    if log_floor is not None
                    else scatter_y_raw
                )
                scatter_x = [
                    _int(row, "M") + x_offset + instance_offset_by_id[_int(row, "instance_id")]
                    for row in row_group
                ]
                if clipped_group and mark_zero_clipped and log_floor is not None:
                    ax.scatter(
                        scatter_x,
                        scatter_y,
                        s=17,
                        facecolors="none",
                        edgecolors=color,
                        marker=marker,
                        alpha=0.9,
                        linewidths=0.65,
                        zorder=4,
                    )
                else:
                    ax.scatter(
                        scatter_x,
                        scatter_y,
                        s=11,
                        color=color,
                        marker=marker,
                        alpha=0.43,
                        linewidths=0.0,
                        zorder=2,
                    )

        median_rows = [row for row in aggregated_rows if model_key in row]
        median_x = [int(row["M"]) + x_offset for row in median_rows]
        median_y_raw = [float(row[model_key]["median"]) for row in median_rows]
        median_y = (
            [max(value, float(log_floor)) for value in median_y_raw]
            if log_floor is not None
            else median_y_raw
        )
        median_m = [int(row["M"]) for row in median_rows]
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


def _clip_negative_gaps_to_zero(
    series_rows: dict[str, Sequence[dict[str, str]]],
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    clipped: dict[str, list[dict[str, str]]] = {}
    for model_key, rows in series_rows.items():
        column = str(model_configs[model_key]["column"])
        zero_flag = f"{ZERO_CLIPPED_FLAG_PREFIX}{column}"
        clipped_rows: list[dict[str, str]] = []
        for row in rows:
            value = _float(row, column)
            next_row = dict(row)
            if value < 0.0:
                next_row[column] = "0.0"
                next_row[zero_flag] = "1"
            else:
                next_row[zero_flag] = "0"
            clipped_rows.append(next_row)
        clipped[model_key] = clipped_rows
    return clipped


def make_two_panel(
    *,
    results_csv: Path,
    degree1_results_csv: Path,
    random_sparse_results_csv: Path | None = None,
    ucr_random_sparse_results_csv: Path,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    omit_walsh_random_sparse: bool = False,
    baseline_color: str | None = None,
    wd1_label: str | None = None,
    wd1_color: str | None = None,
    ucr_random_sparse_label: str | None = None,
    ucr_random_sparse_color: str | None = None,
    show_ucr_random_sparse_instances: bool = False,
    clip_negative_gaps_to_zero: bool = False,
    mark_zero_clipped: bool = False,
) -> dict[str, Any]:
    config_kwargs = {
        "omit_walsh_random_sparse": bool(omit_walsh_random_sparse),
        "baseline_color": baseline_color,
        "wd1_label": wd1_label,
        "wd1_color": wd1_color,
        "ucr_random_sparse_label": ucr_random_sparse_label,
        "ucr_random_sparse_color": ucr_random_sparse_color,
    }
    detail_configs = _model_configs("detail", **config_kwargs)
    all_configs = _model_configs("all", **config_kwargs)
    input_paths = {
        "full": results_csv,
        "degree1": degree1_results_csv,
        "random_sparse": random_sparse_results_csv,
        "ucr_random_sparse": ucr_random_sparse_results_csv,
    }
    used_sources = sorted(
        {str(config["source"]) for config in [*detail_configs.values(), *all_configs.values()]}
    )
    missing_paths = [source for source in used_sources if input_paths[source] is None]
    if missing_paths:
        raise ValueError(f"Missing required input path(s): {missing_paths}")
    source_rows = {source: _read_rows(input_paths[source]) for source in used_sources}
    empty_sources = [
        (source, path)
        for source, path in input_paths.items()
        if source in used_sources
        if not source_rows[source]
    ]
    if empty_sources:
        details = ", ".join(f"{source}={path}" for source, path in empty_sources)
        raise ValueError(f"No rows found for: {details}")

    detail_series_rows = _series_rows(
        source_rows=source_rows,
        model_configs=detail_configs,
    )
    all_series_rows = _series_rows(
        source_rows=source_rows,
        model_configs=all_configs,
    )
    if clip_negative_gaps_to_zero:
        detail_series_rows = _clip_negative_gaps_to_zero(detail_series_rows, detail_configs)
        all_series_rows = _clip_negative_gaps_to_zero(all_series_rows, all_configs)
    detail_aggregated = _aggregate_by_m(detail_series_rows, detail_configs)
    all_aggregated = _aggregate_by_m(all_series_rows, all_configs)
    m_values = sorted({_int(row, "M") for row in _all_rows(all_series_rows)})

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "lines.linewidth": 1.1,
        }
    )
    fig, (ax_detail, ax_full) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.5),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(wspace=0.065)

    _plot_panel(
        ax_detail,
        series_rows=detail_series_rows,
        aggregated_rows=detail_aggregated,
        model_configs=detail_configs,
        show_instances=True,
        log_floor=DETAIL_LOG_YLIM[0],
        mark_zero_clipped=bool(mark_zero_clipped),
    )
    _style_axes(ax_detail, m_values=m_values)
    ax_detail.set_yscale("log")
    ax_detail.set_ylim(*DETAIL_LOG_YLIM)
    ax_detail.grid(True, which="minor", axis="y", color="0.92", linewidth=0.35)
    _add_regime_span(ax_detail, x0=4.65, x1=8.35, label="Easy regime", y=1.035)
    _add_regime_span(ax_detail, x0=8.65, x1=12.35, label="Hard regime", y=1.035)
    ax_detail.text(
        -0.09,
        1.03,
        "(a)",
        transform=ax_detail.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        clip_on=False,
    )

    _plot_panel(
        ax_full,
        series_rows=all_series_rows,
        aggregated_rows=all_aggregated,
        model_configs=all_configs,
        show_instances=False,
        instance_model_keys={"ucr_random_sparse"} if show_ucr_random_sparse_instances else None,
        log_floor=None,
        mark_zero_clipped=False,
    )
    _style_axes(ax_full, m_values=m_values)
    ax_full.set_ylabel("")
    ax_full.yaxis.tick_right()
    ax_full.yaxis.set_label_position("right")
    ax_full.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax_full.text(
        0.0,
        1.03,
        "(b)",
        transform=ax_full.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        clip_on=False,
    )
    ax_full.legend(
        frameon=False,
        handlelength=1.25,
        columnspacing=0.65,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=3,
        borderaxespad=0.0,
    )

    zoom_box = Rectangle(
        (ZOOM_BOX_XLIM[0], DETAIL_YLIM[0]),
        ZOOM_BOX_XLIM[1] - ZOOM_BOX_XLIM[0],
        DETAIL_YLIM[1] - DETAIL_YLIM[0],
        fill=False,
        edgecolor=CONNECTOR_COLOR,
        linewidth=0.85,
        linestyle=(0, (3, 2)),
        zorder=5,
    )
    ax_full.add_patch(zoom_box)

    for y_value, detail_y in ((DETAIL_YLIM[1], 1.0), (DETAIL_YLIM[0], 0.0)):
        connector = ConnectionPatch(
            xyA=(ZOOM_BOX_XLIM[0], y_value),
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
        "input_csvs": {
            source: str(input_paths[source])
            for source in ("full", "degree1", "random_sparse", "ucr_random_sparse")
            if source in used_sources and input_paths[source] is not None
        },
        "artifacts": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "summary_json": str(summary_path),
        "panel_a": {
            "description": "Detail view with individual instances and median trends.",
            "models": list(detail_configs),
            "yscale": "log",
            "ylim": list(DETAIL_LOG_YLIM),
            "display_floor": DETAIL_LOG_YLIM[0],
        },
        "panel_b": {
            "description": (
                "Full-scale median plot excluding Walsh random sparse."
                if omit_walsh_random_sparse
                else "Full-scale median plot including Walsh and full-UCR random sparse."
            ),
            "models": list(all_configs),
            "zoom_box": {
                "xlim": list(ZOOM_BOX_XLIM),
                "ylim": list(DETAIL_YLIM),
            },
        },
        "display": {
            "omit_walsh_random_sparse": bool(omit_walsh_random_sparse),
            "baseline_color": baseline_color,
            "wd1_label": wd1_label,
            "wd1_color": wd1_color,
            "ucr_random_sparse_label": ucr_random_sparse_label,
            "ucr_random_sparse_color": ucr_random_sparse_color,
            "show_ucr_random_sparse_instances": bool(show_ucr_random_sparse_instances),
        },
        "gap_display": {
            "clip_negative_gaps_to_zero": bool(clip_negative_gaps_to_zero),
            "mark_zero_clipped": bool(mark_zero_clipped),
            "zero_display_floor": DETAIL_LOG_YLIM[0] if clip_negative_gaps_to_zero else None,
            "note": (
                "Negative raw gaps are replaced by 0 before aggregation and plotting; "
                "zeros in the log-scale detail panel are drawn at the display floor."
                if clip_negative_gaps_to_zero
                else "Raw gaps are used; log-scale detail panel applies only a display floor."
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-panel full-UCR, Walsh degree-1, and Walsh random-sparse gap figure."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--degree1-results-csv",
        type=Path,
        default=DEFAULT_DEGREE1_RESULTS_CSV,
    )
    parser.add_argument(
        "--random-sparse-results-csv",
        type=Path,
        default=DEFAULT_RANDOM_SPARSE_RESULTS_CSV,
    )
    parser.add_argument(
        "--ucr-random-sparse-results-csv",
        type=Path,
        default=DEFAULT_UCR_RANDOM_SPARSE_RESULTS_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--omit-walsh-random-sparse",
        action="store_true",
        help="Exclude the Walsh random sparse curve from both panels.",
    )
    parser.add_argument(
        "--baseline-color",
        type=str,
        default=None,
        help="Override the full-UCR baseline color, e.g. '0.20' or 'black'.",
    )
    parser.add_argument(
        "--wd1-label",
        type=str,
        default=None,
        help="Override the Walsh degree-1 legend label.",
    )
    parser.add_argument(
        "--wd1-color",
        type=str,
        default=None,
        help="Override the Walsh degree-1 color.",
    )
    parser.add_argument(
        "--ucr-random-sparse-label",
        type=str,
        default=None,
        help="Override the full-UCR random sparse legend label.",
    )
    parser.add_argument(
        "--ucr-random-sparse-color",
        type=str,
        default=None,
        help="Override the full-UCR random sparse color.",
    )
    parser.add_argument(
        "--show-ucr-random-sparse-instances",
        action="store_true",
        help="Show full-UCR random sparse per-instance points in the right panel.",
    )
    parser.add_argument(
        "--clip-negative-gaps-to-zero",
        action="store_true",
        help=(
            "Replace negative gaps by zero before aggregation and plotting. "
            "Useful when negative gaps are treated as numerical tolerance artifacts."
        ),
    )
    parser.add_argument(
        "--mark-zero-clipped",
        action="store_true",
        help="Draw raw-negative points with open markers at the log-scale display floor.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = make_two_panel(
        results_csv=args.results_csv.expanduser().resolve(),
        degree1_results_csv=args.degree1_results_csv.expanduser().resolve(),
        random_sparse_results_csv=args.random_sparse_results_csv.expanduser().resolve(),
        ucr_random_sparse_results_csv=args.ucr_random_sparse_results_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        output_stem=str(args.output_stem),
        dpi=int(args.dpi),
        omit_walsh_random_sparse=bool(args.omit_walsh_random_sparse),
        baseline_color=args.baseline_color,
        wd1_label=args.wd1_label,
        wd1_color=args.wd1_color,
        ucr_random_sparse_label=args.ucr_random_sparse_label,
        ucr_random_sparse_color=args.ucr_random_sparse_color,
        show_ucr_random_sparse_instances=bool(args.show_ucr_random_sparse_instances),
        clip_negative_gaps_to_zero=bool(args.clip_negative_gaps_to_zero),
        mark_zero_clipped=bool(args.mark_zero_clipped),
    )
    print(f"saved: {summary['artifacts']['png']}")
    print(f"saved: {summary['artifacts']['pdf']}")
    print(f"saved: {summary['summary_json']}")


if __name__ == "__main__":
    main()
