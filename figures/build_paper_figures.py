from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "paper"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "paper"
IMPL_DIR = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl"
DEFAULT_FORMATS = ("pdf", "png")

WH_D8_STEM = "wh_md_sweep_gap_zoom_two_panel_no_walsh_rs_wd1_rsucr_median_only"
HAAR_D8_STEM = "exact_haar_d8_gap_single_panel"
WH_DEGREE_STEM = "wh_md_walsh_degree_gap_instances_median_tableau4_k1_k5_full_compressed_opaque_lw06"


def _load_module(module_name: str, module_path: Path, *, import_root: Path | None = None) -> Any:
    resolved = module_path.resolve()
    added = False
    if import_root is not None:
        root = str(import_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
            added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {module_name} from {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            try:
                sys.path.remove(str(import_root.resolve()))
            except ValueError:
                pass


def _figure_paths(output_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    return [output_dir / f"{stem}.{fmt}" for fmt in formats]


def plot_fig_wh_d8_sweep(
    data_root: Path,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    plot_dir = IMPL_DIR
    module = _load_module(
        "walshucr_plot_wh_md_gap_zoom_two_panel",
        plot_dir / "plots/fig_wh.py",
        import_root=plot_dir,
    )
    raw = data_root / "fig_wh_d8_sweep" / "raw"
    module.make_two_panel(
        results_csv=(raw / "wh_md_sweep_results.csv").resolve(),
        degree1_results_csv=(raw / "wh_md_walsh_degree1_results.csv").resolve(),
        ucr_random_sparse_results_csv=(raw / "random_sparse_vs_degree1_results.csv").resolve(),
        output_dir=output_dir,
        output_stem=WH_D8_STEM,
        dpi=300,
        omit_walsh_random_sparse=True,
        baseline_color="tab:blue",
        wd1_label="WD-1",
        wd1_color="tab:orange",
        ucr_random_sparse_label="RS-UCR",
        ucr_random_sparse_color="tab:red",
    )
    return _figure_paths(output_dir, WH_D8_STEM, formats)


def plot_fig_haar_d8_sweep(
    data_root: Path,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    module = _load_module(
        "walshucr_plot_haar_d8_exact_gap_zoom_two_panel",
        IMPL_DIR / "plots" / "fig_haar.py",
        import_root=IMPL_DIR,
    )
    module.make_single_panel(
        results_csv=(
            data_root / "fig_haar_d8_sweep" / "raw" / "exact_haar_d8_sweep_results.csv"
        ).resolve(),
        output_dir=output_dir,
        output_stem=HAAR_D8_STEM,
        dpi=300,
    )
    return _figure_paths(output_dir, HAAR_D8_STEM, formats)


def plot_fig_wh_degree_sweep(data_root: Path, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    module = _load_module(
        "walshucr_plot_wh_md_walsh_degree_gap_instances_median",
        IMPL_DIR / "plots/fig_wh_sweep.py",
        import_root=IMPL_DIR,
    )
    module.make_plot(
        results_csv=(
            data_root / "fig_wh_degree_sweep" / "raw" / "wh_md_walsh_degree_sweep_results.csv"
        ).resolve(),
        output_dir=output_dir,
        output_stem=WH_DEGREE_STEM,
        dpi=300,
        numerical_floor=3.0e-5,
    )
    return _figure_paths(output_dir, WH_DEGREE_STEM, formats)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_figure_manifest(paths: Sequence[Path], output_dir: Path) -> None:
    records = []
    for path in sorted(paths):
        try:
            display_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            display_path = path.as_posix()
        records.append(
            {
                "path": display_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (output_dir / "figure_manifest.json").write_text(
        json.dumps({"manifest_version": 1, "files": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build paper figure files from data/paper.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["fig_wh_d8_sweep", "fig_haar_d8_sweep", "fig_wh_degree_sweep"],
        default=["fig_wh_d8_sweep", "fig_haar_d8_sweep", "fig_wh_degree_sweep"],
    )
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=["pdf", "png"])
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    formats = list(dict.fromkeys(str(fmt) for fmt in args.formats))

    generated: list[Path] = []
    if "fig_wh_d8_sweep" in args.figures:
        generated.extend(plot_fig_wh_d8_sweep(data_root, output_dir, formats))
    if "fig_haar_d8_sweep" in args.figures:
        generated.extend(plot_fig_haar_d8_sweep(data_root, output_dir, formats))
    if "fig_wh_degree_sweep" in args.figures:
        generated.extend(plot_fig_wh_degree_sweep(data_root, output_dir, formats))
    _write_figure_manifest(generated, output_dir)
    for path in generated:
        print(f"saved: {path}")
    print(f"manifest: {output_dir / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
