from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMPL_DIR = Path(__file__).resolve().parents[1]
if str(IMPL_DIR) not in sys.path:
    sys.path.append(str(IMPL_DIR))

from common import _float, _int, _read_rows


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
DEFAULT_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_degree_sweep"
    / "raw"
    / "wh_md_walsh_degree_sweep_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_CSV.parent.parent / "figures"
DEFAULT_OUTPUT_STEM = "wh_md_walsh_degree_gap_instances_median_tableau4_k1_k5_full_compressed_opaque_lw06"
DEFAULT_NUMERICAL_FLOOR = 3.0e-5


def _clip(value: float, floor: float) -> float:
    return max(float(value), float(floor))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of an empty sequence.")
    return float(statistics.median(values))


def _rows_for(rows: Sequence[dict[str, str]], *, m_value: int, instance_id: int | None = None) -> list[dict[str, str]]:
    selected = [row for row in rows if _int(row, "M") == int(m_value)]
    if instance_id is not None:
        selected = [row for row in selected if _int(row, "instance_id") == int(instance_id)]
    return sorted(selected, key=lambda row: _int(row, "degree"))


def _median_curve(
    rows: Sequence[dict[str, str]],
    *,
    m_value: int,
    degrees: Sequence[int],
    full_x: int,
    floor: float,
) -> tuple[list[int], list[float]]:
    rows_m = _rows_for(rows, m_value=m_value)
    y_values: list[float] = []
    for degree in degrees:
        y_values.append(
            _median(
                [
                    _clip(_float(row, "gap_abs"), floor)
                    for row in rows_m
                    if _int(row, "degree") == int(degree)
                ]
            )
        )
    full_by_instance = {
        _int(row, "instance_id"): _clip(_float(row, "gap_abs_full_ref"), floor)
        for row in rows_m
    }
    y_values.append(_median(list(full_by_instance.values())))
    return [*degrees, int(full_x)], y_values


def make_plot(
    *,
    results_csv: Path,
    output_dir: Path,
    output_stem: str,
    dpi: int,
    numerical_floor: float = DEFAULT_NUMERICAL_FLOOR,
) -> dict[str, Any]:
    rows = _read_rows(results_csv)
    if not rows:
        raise ValueError(f"No rows found in {results_csv}.")
    required_columns = {"M", "instance_id", "degree", "gap_abs", "gap_abs_full_ref"}
    missing = sorted(required_columns.difference(rows[0]))
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")

    m_values = sorted({_int(row, "M") for row in rows})
    degrees = sorted({_int(row, "degree") for row in rows})
    full_x = max(degrees) + 1
    floor = float(numerical_floor)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(m_values)]

    plt.rcParams.update(
        {
            "font.size": 16.0,
            "axes.labelsize": 22.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 18.0,
            "legend.fontsize": 15.0,
            "legend.title_fontsize": 17.0,
            "lines.linewidth": 1.2,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.15))

    median_summary: list[dict[str, Any]] = []
    for idx, m_value in enumerate(m_values):
        color = colors[idx % len(colors)]
        rows_m = _rows_for(rows, m_value=m_value)
        instance_ids = sorted({_int(row, "instance_id") for row in rows_m})
        for instance_id in instance_ids:
            bucket = _rows_for(rows, m_value=m_value, instance_id=instance_id)
            x_values = [_int(row, "degree") for row in bucket] + [full_x]
            y_values = [_clip(_float(row, "gap_abs"), floor) for row in bucket]
            y_values.append(_clip(_float(bucket[0], "gap_abs_full_ref"), floor))
            ax.plot(
                x_values,
                y_values,
                color=color,
                marker="o",
                linewidth=0.6,
                markersize=2.8,
                alpha=1.0,
                zorder=2,
            )

        median_x, median_y = _median_curve(
            rows,
            m_value=m_value,
            degrees=degrees,
            full_x=full_x,
            floor=floor,
        )
        ax.plot(
            median_x,
            median_y,
            color=color,
            marker="o",
            linewidth=2.9,
            markersize=7.4,
            label=f"M={m_value}",
            zorder=4,
        )
        median_summary.append(
            {
                "M": int(m_value),
                "x": [*degrees, "full-UCR"],
                "median_gap_abs_clipped": median_y,
            }
        )

    ax.set_xlabel("Walsh degree")
    ax.set_ylabel(r"$\Delta_{\mathrm{opt}}$")
    ax.set_xticks([*degrees, full_x])
    ax.set_xticklabels([str(degree) for degree in degrees] + ["full-UCR"])
    ax.set_yscale("log")
    ax.axhspan(floor / 1.4, floor, color="#d8d8d8", alpha=0.45, zorder=0)
    ax.axhline(floor, color="0.50", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_ylim(bottom=floor / 1.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        title="median over instances",
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        fancybox=True,
        edgecolor="0.75",
        facecolor="white",
        handlelength=1.45,
        borderpad=0.35,
        labelspacing=0.6,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.985)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    summary_path = output_dir / f"{output_stem}_summary.json"
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "input_csvs": {"wh_degree_sweep": str(results_csv)},
        "row_count": len(rows),
        "m_values": m_values,
        "degrees": degrees,
        "instance_ids": sorted({_int(row, "instance_id") for row in rows}),
        "numerical_floor": floor,
        "artifacts": {"png": str(png_path), "pdf": str(pdf_path)},
        "summary_json": str(summary_path),
        "plot_design": {
            "description": (
                "Data-based reconstruction of the paper WH Walsh-degree sweep: "
                "per-instance trajectories plus median curves over instances."
            ),
            "x_axis": "Walsh degree 1..5 plus full-UCR reference",
            "y_axis": "log clipped optimum gap",
            "colors": "Matplotlib Tableau first four colors for M=9,10,11,12",
            "instance_linewidth": 0.6,
            "median_linewidth": 2.9,
        },
        "median_curves": median_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the WH Walsh-degree sweep as instance trajectories plus medians."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=str, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--numerical-floor", type=float, default=DEFAULT_NUMERICAL_FLOOR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = make_plot(
        results_csv=args.results_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        output_stem=str(args.output_stem),
        dpi=int(args.dpi),
        numerical_floor=float(args.numerical_floor),
    )
    print(f"saved: {summary['artifacts']['png']}")
    print(f"saved: {summary['artifacts']['pdf']}")
    print(f"saved: {summary['summary_json']}")


if __name__ == "__main__":
    main()
