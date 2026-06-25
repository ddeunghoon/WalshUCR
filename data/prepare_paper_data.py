from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PAPER_DATA_DIR = ROOT / "data" / "paper"
MANIFEST_PATH = ROOT / "data" / "manifests" / "paper_data_manifest.json"
D16_OUTPUT_DIR = PAPER_DATA_DIR / "table_d16_checks"


PAPER_FILES: list[dict[str, str]] = [
    {
        "artifact": "fig_wh_d8_sweep",
        "role": "full_ucr_reference_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/results/wh_md_sweep_nsys3_scale1_drop_extra_restart_reuse_i10_r50/final/raw/wh_md_sweep_results.csv",
        "dest": "data/paper/fig_wh_d8_sweep/raw/wh_md_sweep_results.csv",
    },
    {
        "artifact": "fig_wh_d8_sweep",
        "role": "walsh_degree1_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/results/wh_md_walsh_degree1_nsys3_scale1_drop_extra_restart_reuse_i10_r50/final/raw/wh_md_walsh_degree1_results.csv",
        "dest": "data/paper/fig_wh_d8_sweep/raw/wh_md_walsh_degree1_results.csv",
    },
    {
        "artifact": "fig_wh_d8_sweep",
        "role": "rs_ucr_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/random_sparse_model/results/random_sparse_ucr_vs_degree1_nsys3_scale1_drop_extra_i10_r50/final/raw/random_sparse_vs_degree1_results.csv",
        "dest": "data/paper/fig_wh_d8_sweep/raw/random_sparse_vs_degree1_results.csv",
    },
    {
        "artifact": "fig_haar_d8_sweep",
        "role": "exact_haar_d8_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/haar_d8/results/exact_haar_d8_i10_r50/final/raw/exact_haar_d8_sweep_results.csv",
        "dest": "data/paper/fig_haar_d8_sweep/raw/exact_haar_d8_sweep_results.csv",
    },
    {
        "artifact": "fig_wh_degree_sweep",
        "role": "walsh_degree_sweep_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/results/wh_md_walsh_degree_sweep_nsys3_M9_M12_drop_extra_random_i5_r100/final/raw/wh_md_walsh_degree_sweep_results.csv",
        "dest": "data/paper/fig_wh_degree_sweep/raw/wh_md_walsh_degree_sweep_results.csv",
    },
    {
        "artifact": "appendix_rank_diagnostics",
        "role": "rank_diagnostics_csv",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/linear_independence_verification/raw/input_ensemble_gram_rank_results.csv",
        "dest": "data/paper/appendix_rank_diagnostics/raw/input_ensemble_gram_rank_results.csv",
    },
    {
        "artifact": "appendix_rank_diagnostics",
        "role": "rank_diagnostics_report",
        "source": "pennylane_qsd/experiments/ucr_method/sec5/linear_independence_verification/reports/input_ensemble_linear_independence_report.md",
        "dest": "data/paper/appendix_rank_diagnostics/reports/input_ensemble_linear_independence_report.md",
    },
]


D16_RESULT_ROOTS = [
    "gpu_qsd/results/memopt_weyl_wd1_fullucr_nsys4_M16_instances00-09_r5_su61_steps1000",
    "gpu_qsd/results/memopt_weyl_wd4_fullucr_nsys4_M17_instances00-09_r2_su61_steps1000_nomemopt",
    "gpu_qsd/results/memopt_weyl_wd4_fullucr_nsys4_M20_instances00-09_r2_su61_steps1000_nomemopt",
    "gpu_qsd/results/memopt_haar_wd1_fullucr_nsys4_M16_instances00-09_r5_su61_steps1000_nomemopt",
    "gpu_qsd/results/memopt_haar_wd4_fullucr_nsys4_M17_instances00-09_r5_su61_steps1000_nomemopt",
    "gpu_qsd/results/memopt_haar_wd4_fullucr_nsys4_M20_instances00-09_r5_su61_steps1000_nomemopt",
]


STATIC_TABLES: dict[str, dict[str, Any]] = {
    "data/paper/table_param_decomposition/table_param_decomposition.csv": {
        "artifact": "table_param_decomposition",
        "rows": [
            {
                "regime": "M<=d",
                "d": 8,
                "M_min": 5,
                "M_max": 8,
                "n_anc": 3,
                "shared_system_only_params": 455,
                "full_ucr_angles": 56,
                "wd1_angles": 15,
                "wd2_angles": 34,
                "wd3_angles": 49,
                "rs_ucr_angles": 15,
            },
            {
                "regime": "M>d",
                "d": 8,
                "M_min": 9,
                "M_max": 12,
                "n_anc": 4,
                "shared_system_only_params": 975,
                "full_ucr_angles": 120,
                "wd1_angles": 22,
                "wd2_angles": 56,
                "wd3_angles": 91,
                "rs_ucr_angles": 22,
            },
        ],
    },
    "data/paper/table_app_ensemble_grid/table_app_ensemble_grid.csv": {
        "artifact": "table_app_ensemble_grid",
        "rows": [
            {
                "dataset": "WH dense sweep",
                "section": "sec:exp-d8-sweeps",
                "figure_or_table": "fig:wh",
                "ensemble_family": "Weyl--Heisenberg",
                "d": 8,
                "n_sys": 3,
                "M_values": "5,6,7,8,9,10,11,12",
                "instances_per_M": 10,
                "models": "full-UCR;WD-1;RS-UCR",
                "restarts_per_instance": 50,
                "steps": 1000,
                "paper_data_dir": "data/paper/fig_wh_d8_sweep",
            },
            {
                "dataset": "Haar dense sweep",
                "section": "sec:exp-d8-sweeps",
                "figure_or_table": "fig:haar",
                "ensemble_family": "Haar-random",
                "d": 8,
                "n_sys": 3,
                "M_values": "5,6,7,8,9,10,11,12",
                "instances_per_M": 10,
                "models": "full-UCR;WD-1;RS-UCR",
                "restarts_per_instance": 50,
                "steps": 1000,
                "paper_data_dir": "data/paper/fig_haar_d8_sweep",
            },
            {
                "dataset": "WH Walsh-degree sweep",
                "section": "sec:exp-degree",
                "figure_or_table": "fig:wh-sweep",
                "ensemble_family": "Weyl--Heisenberg",
                "d": 8,
                "n_sys": 3,
                "M_values": "9,10,11,12",
                "instances_per_M": 5,
                "models": "WD-1;WD-2;WD-3;WD-4;WD-5",
                "restarts_per_instance": 100,
                "steps": 1000,
                "paper_data_dir": "data/paper/fig_wh_degree_sweep",
            },
            {
                "dataset": "WH d=16 checks",
                "section": "sec:exp-d16",
                "figure_or_table": "tab:d16-checks",
                "ensemble_family": "Weyl--Heisenberg",
                "d": 16,
                "n_sys": 4,
                "M_values": "16,17,20",
                "instances_per_M": 10,
                "models": "full-UCR;WD-1 for M=16;WD-4 for M=17,20",
                "restarts_per_instance": 5,
                "steps": 1000,
                "paper_data_dir": "data/paper/table_d16_checks",
            },
            {
                "dataset": "Haar d=16 checks",
                "section": "sec:exp-d16",
                "figure_or_table": "tab:d16-checks",
                "ensemble_family": "Haar-random",
                "d": 16,
                "n_sys": 4,
                "M_values": "16,17,20",
                "instances_per_M": 10,
                "models": "full-UCR;WD-1 for M=16;WD-4 for M=17,20",
                "restarts_per_instance": 5,
                "steps": 1000,
                "paper_data_dir": "data/paper/table_d16_checks",
            },
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row_count = sum(1 for _ in reader)
        return {
            "format": "csv",
            "row_count": row_count,
            "columns": list(reader.fieldnames or []),
        }


def _jsonl_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        record_count = sum(1 for line in handle if line.strip())
    return {"format": "jsonl", "record_count": record_count}


def _json_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema: dict[str, Any] = {"format": "json", "top_level_type": type(payload).__name__}
    if isinstance(payload, dict):
        schema["top_level_keys"] = sorted(str(key) for key in payload)
    return schema


def _file_schema(path: Path) -> dict[str, Any]:
    if path.suffix == ".csv":
        return _csv_schema(path)
    if path.suffix == ".jsonl":
        return _jsonl_schema(path)
    if path.suffix == ".json":
        return _json_schema(path)
    return {"format": path.suffix.lstrip(".") or "text"}


def _copy_paper_data_files() -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for item in PAPER_FILES:
        source = PROJECT_ROOT / item["source"]
        dest = ROOT / item["dest"]
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(item)
    return copied


def _load_aggregate_table_module() -> Any:
    path = ROOT / "experiments" / "sec5_numerical_experiments" / "table_d16_checks" / "aggregate_table.py"
    spec = importlib.util.spec_from_file_location("walshucr_aggregate_table", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load aggregate_table module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate_d16() -> None:
    roots = [PROJECT_ROOT / rel_path for rel_path in D16_RESULT_ROOTS]
    missing = [path for path in roots if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing d=16 result roots:\n" + "\n".join(str(path) for path in missing))
    module = _load_aggregate_table_module()
    module.main(["--input-roots", *[str(path) for path in roots], "--output-dir", str(D16_OUTPUT_DIR)])


def _write_static_tables() -> None:
    for rel_path, payload in STATIC_TABLES.items():
        rows = payload["rows"]
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _remove_stale_readmes() -> None:
    for path in PAPER_DATA_DIR.rglob("README.md"):
        path.unlink()


def _artifact_for(path: Path) -> str:
    relative = path.relative_to(PAPER_DATA_DIR)
    if len(relative.parts) == 1:
        return "paper_data_root"
    return relative.parts[0] if relative.parts else ""


def _source_index(copied: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {item["dest"]: item for item in copied}


def _write_manifest(copied: Sequence[dict[str, str]]) -> None:
    source_index = _source_index(copied)
    records: list[dict[str, Any]] = []
    for path in sorted(PAPER_DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        rel_path = path.relative_to(ROOT).as_posix()
        source_item = source_index.get(rel_path, {})
        records.append(
            {
                "path": rel_path,
                "artifact": source_item.get("artifact", _artifact_for(path)),
                "role": source_item.get("role", ""),
                "source": source_item.get("source", ""),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "schema": _file_schema(path),
            }
        )
    manifest = {
        "manifest_version": 1,
        "description": "Paper data for WalshUCR figures and tables.",
        "files": records,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare local canonical paper data under data/paper.")
    parser.add_argument("--skip-d16", action="store_true", help="Do not aggregate the d=16 GPU-result roots.")
    args = parser.parse_args(argv)

    copied = _copy_paper_data_files()
    _write_static_tables()
    if not args.skip_d16:
        _aggregate_d16()
    _remove_stale_readmes()
    _write_manifest(copied)

    print(f"copied_files={len(copied)}")
    print(f"paper_data_files={sum(1 for path in PAPER_DATA_DIR.rglob('*') if path.is_file())}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
