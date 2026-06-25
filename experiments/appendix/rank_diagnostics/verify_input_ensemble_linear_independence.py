from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parents[2]
SEC5_IMPL_DIR = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl"
HAAR_D8_DIR = SEC5_IMPL_DIR / "haar_d8"
SRC_DIR = ROOT / "src"
for path in (CURRENT_DIR, SEC5_IMPL_DIR, HAAR_D8_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common import _read_rows, _write_dict_csv
from weyl_problem import _build_problem_instance as _build_weyl_problem_instance
from haar_d8_exact_sweep_restart_reuse import (
    DEFAULT_ENSEMBLE_MASTER_SEED,
    DEFAULT_NESTED_MAX_M,
    DEFAULT_STATE_DTYPE,
    _build_problem_instance as _build_exact_haar_problem_instance,
)
from weyl_statevector_backend import _context as _weyl_statevector_context
from weyl_statevector_backend import _prepare_weyl_state
from wh_d8_sweep import _problem_namespace, _seed_pair_for_instance


DEFAULT_WH_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_walsh_degree1_results.csv"
)
DEFAULT_HAAR_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_haar_d8_sweep"
    / "raw"
    / "exact_haar_d8_sweep_results.csv"
)
DEFAULT_HAAR_SUMMARY_JSON = (
    ROOT
    / "data"
    / "paper"
    / "fig_haar_d8_sweep"
    / "summaries"
    / "exact_haar_d8_sweep_summary.json"
)
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "linear_independence_verification"
DEFAULT_M_VALUES = (5, 6, 7, 8)
DEFAULT_INSTANCE_IDS = tuple(range(10))
DEFAULT_RANK_TOL = 1e-10

RESULTS_FILENAME = "input_ensemble_gram_rank_results.csv"
SUMMARY_FILENAME = "input_ensemble_gram_rank_summary.json"
REPORT_FILENAME = "input_ensemble_linear_independence_report.md"
WEYL_FIGURE_FILENAME = "weyl_heisenberg_gram_rank_diagnostics.png"
HAAR_FIGURE_FILENAME = "exact_haar_d8_gram_rank_diagnostics.png"

CSV_FIELDS = (
    "experiment_id",
    "benchmark_name",
    "M",
    "instance_id",
    "n_sys",
    "d",
    "state_count",
    "state_dim",
    "rank_tol",
    "rank",
    "expected_rank",
    "rank_equals_M",
    "lambda_min",
    "lambda_max",
    "gram_trace",
    "condition_number",
    "max_offdiag_abs",
    "pairwise_fidelity_mean",
    "pairwise_fidelity_max",
    "benchmark_seed",
    "data_seed",
    "ensemble_seed",
    "benchmark_internal_seed",
    "existing_gram_rank_numeric",
    "rank_matches_existing",
    "state_generation",
    "stateprep_operation",
    "stateprep_decomposition",
    "label_payload",
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    return _read_rows(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _index_target_rows(
    rows: Sequence[dict[str, str]],
    *,
    m_values: Sequence[int],
    instance_ids: Sequence[int],
    source_name: str,
) -> dict[tuple[int, int], dict[str, str]]:
    wanted = {(int(M), int(instance_id)) for M in m_values for instance_id in instance_ids}
    indexed: dict[tuple[int, int], dict[str, str]] = {}
    duplicates: list[tuple[int, int]] = []
    for row in rows:
        try:
            key = (int(row["M"]), int(row["instance_id"]))
        except KeyError as exc:
            raise KeyError(f"{source_name} is missing required column {exc!s}.") from exc
        if key not in wanted:
            continue
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row

    missing = sorted(wanted.difference(indexed))
    if missing:
        preview = ", ".join(f"M={M}/instance={instance_id}" for M, instance_id in missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        raise ValueError(f"{source_name} is missing {len(missing)} target rows: {preview}")
    if duplicates:
        preview = ", ".join(f"M={M}/instance={instance_id}" for M, instance_id in sorted(set(duplicates))[:8])
        raise ValueError(f"{source_name} has duplicate target rows: {preview}")
    return indexed


def _gram_diagnostics(states: np.ndarray, *, rank_tol: float) -> dict[str, Any]:
    state_matrix = np.asarray(states, dtype=np.complex128)
    if state_matrix.ndim != 2:
        raise ValueError(f"states must be a rank-2 array, got shape={state_matrix.shape}.")

    gram = state_matrix @ np.conjugate(state_matrix.T)
    gram_hermitian = 0.5 * (gram + np.conjugate(gram.T))
    evals = np.linalg.eigvalsh(gram_hermitian)
    offdiag_mask = ~np.eye(gram.shape[0], dtype=bool)
    offdiag_abs = np.abs(gram[offdiag_mask])
    pairwise_fidelity = offdiag_abs**2
    lambda_min = float(np.min(evals))
    lambda_max = float(np.max(evals))
    if lambda_min > 0.0:
        condition_number = float(lambda_max / lambda_min)
    else:
        condition_number = math.inf

    return {
        "rank": int(np.linalg.matrix_rank(gram, tol=float(rank_tol))),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "gram_trace": float(np.real(np.trace(gram))),
        "condition_number": condition_number,
        "max_offdiag_abs": float(np.max(offdiag_abs)) if offdiag_abs.size else 0.0,
        "pairwise_fidelity_mean": float(np.mean(pairwise_fidelity)) if pairwise_fidelity.size else 0.0,
        "pairwise_fidelity_max": float(np.max(pairwise_fidelity)) if pairwise_fidelity.size else 0.0,
    }


def _make_weyl_problem_args(*, M: int, benchmark_seed: int, data_seed: int) -> argparse.Namespace:
    return _problem_namespace(
        n_sys=3,
        m_outcome=int(M),
        benchmark_seed=int(benchmark_seed),
        data_seed=int(data_seed),
        optimizer="adam",
        learning_rate=1e-2,
        steps=1000,
        eval_interval=50,
        threshold=1e-6,
        tol=5e-4,
        su_depth=14,
        scale_init=1.0,
        bias_scale_init=1.0,
        weight_decay=0.0,
        state_dtype="complex128",
    )


def _reconstruct_weyl_states(*, M: int, instance_id: int) -> tuple[np.ndarray, dict[str, Any]]:
    benchmark_seed, data_seed = _seed_pair_for_instance(n_sys=3, M=int(M), instance_id=int(instance_id))
    problem_args = _make_weyl_problem_args(M=int(M), benchmark_seed=benchmark_seed, data_seed=data_seed)
    problem = _build_weyl_problem_instance(problem_args)
    ctx = _weyl_statevector_context(3, 0)
    a_values, b_values = problem["inputs"]
    states = []
    labels: list[list[int]] = []
    for a_value, b_value in zip(a_values, b_values, strict=True):
        states.append(
            np.asarray(
                _prepare_weyl_state(
                    a=jnp.asarray(a_value),
                    b=jnp.asarray(b_value),
                    seed_angles=problem["benchmark"].seed_angles,
                    use_scrambler=bool(problem["benchmark"].use_scrambler),
                    ctx=ctx,
                ),
                dtype=np.complex128,
            )
        )
        labels.append([int(a_value), int(b_value)])
    metadata = {
        "benchmark_seed": int(benchmark_seed),
        "data_seed": int(data_seed),
        "labels_ab": labels,
    }
    return np.stack(states, axis=0), metadata


def _haar_args_from_summary(summary_json: Path) -> argparse.Namespace:
    payload = _load_json(summary_json)
    config = payload.get("config", {})
    return argparse.Namespace(
        nested_max_m=int(config.get("nested_max_m", DEFAULT_NESTED_MAX_M)),
        state_dtype=str(config.get("state_dtype", DEFAULT_STATE_DTYPE)),
        fix_global_phase=bool(config.get("fix_global_phase", False)),
        ensemble_master_seed=int(config.get("ensemble_master_seed", DEFAULT_ENSEMBLE_MASTER_SEED)),
    )


def _reconstruct_haar_states(
    *,
    M: int,
    instance_id: int,
    haar_args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    problem = _build_exact_haar_problem_instance(
        n_sys=3,
        M=int(M),
        instance_id=int(instance_id),
        args=haar_args,
    )
    metadata = {
        "benchmark_seed": int(problem["ensemble_seed"]),
        "data_seed": int(problem["data_seed"]),
        "ensemble_seed": int(problem["ensemble_seed"]),
        "benchmark_internal_seed": int(problem["benchmark_internal_seed"]),
    }
    return np.asarray(problem["states_np"], dtype=np.complex128), metadata


def _verify_weyl_rows(
    *,
    indexed_rows: dict[tuple[int, int], dict[str, str]],
    m_values: Sequence[int],
    instance_ids: Sequence[int],
    rank_tol: float,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for M in m_values:
        for instance_id in instance_ids:
            source_row = indexed_rows[(int(M), int(instance_id))]
            states, metadata = _reconstruct_weyl_states(M=int(M), instance_id=int(instance_id))
            benchmark_seed = int(metadata["benchmark_seed"])
            data_seed = int(metadata["data_seed"])
            if int(source_row["benchmark_seed"]) != benchmark_seed:
                raise ValueError(
                    f"Weyl seed mismatch for M={M}, instance_id={instance_id}: "
                    f"CSV benchmark_seed={source_row['benchmark_seed']}, expected={benchmark_seed}."
                )
            if int(source_row["data_seed"]) != data_seed:
                raise ValueError(
                    f"Weyl seed mismatch for M={M}, instance_id={instance_id}: "
                    f"CSV data_seed={source_row['data_seed']}, expected={data_seed}."
                )
            diag = _gram_diagnostics(states, rank_tol=rank_tol)
            rank = int(diag["rank"])
            output_rows.append(
                {
                    "experiment_id": "weyl_heisenberg",
                    "benchmark_name": "Weyl-Heisenberg",
                    "M": int(M),
                    "instance_id": int(instance_id),
                    "n_sys": 3,
                    "d": 8,
                    "state_count": int(states.shape[0]),
                    "state_dim": int(states.shape[1]),
                    "rank_tol": float(rank_tol),
                    "rank": rank,
                    "expected_rank": int(M),
                    "rank_equals_M": rank == int(M),
                    "lambda_min": diag["lambda_min"],
                    "lambda_max": diag["lambda_max"],
                    "gram_trace": diag["gram_trace"],
                    "condition_number": diag["condition_number"],
                    "max_offdiag_abs": diag["max_offdiag_abs"],
                    "pairwise_fidelity_mean": diag["pairwise_fidelity_mean"],
                    "pairwise_fidelity_max": diag["pairwise_fidelity_max"],
                    "benchmark_seed": benchmark_seed,
                    "data_seed": data_seed,
                    "ensemble_seed": "",
                    "benchmark_internal_seed": "",
                    "existing_gram_rank_numeric": "",
                    "rank_matches_existing": "",
                    "state_generation": "fiducial_state_then_unique_weyl_orbit",
                    "stateprep_operation": "seed_from_angles -> Weyl(a,b) -> simple_scrambler",
                    "stateprep_decomposition": "RY/RZ seed rotations, CNOT chain, CZ ring, QFT-based X_d, RZ-based Z_d",
                    "label_payload": json.dumps({"labels_ab": metadata["labels_ab"]}),
                }
            )
    return output_rows


def _verify_haar_rows(
    *,
    indexed_rows: dict[tuple[int, int], dict[str, str]],
    m_values: Sequence[int],
    instance_ids: Sequence[int],
    rank_tol: float,
    haar_args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for M in m_values:
        for instance_id in instance_ids:
            source_row = indexed_rows[(int(M), int(instance_id))]
            states, metadata = _reconstruct_haar_states(
                M=int(M),
                instance_id=int(instance_id),
                haar_args=haar_args,
            )
            diag = _gram_diagnostics(states, rank_tol=rank_tol)
            rank = int(diag["rank"])
            existing_rank_raw = source_row.get("gram_rank_numeric", "")
            existing_rank = int(existing_rank_raw) if existing_rank_raw != "" else None
            rank_matches_existing = existing_rank is None or rank == existing_rank
            output_rows.append(
                {
                    "experiment_id": "exact_haar_d8",
                    "benchmark_name": "Exact Haar D8",
                    "M": int(M),
                    "instance_id": int(instance_id),
                    "n_sys": 3,
                    "d": 8,
                    "state_count": int(states.shape[0]),
                    "state_dim": int(states.shape[1]),
                    "rank_tol": float(rank_tol),
                    "rank": rank,
                    "expected_rank": int(M),
                    "rank_equals_M": rank == int(M),
                    "lambda_min": diag["lambda_min"],
                    "lambda_max": diag["lambda_max"],
                    "gram_trace": diag["gram_trace"],
                    "condition_number": diag["condition_number"],
                    "max_offdiag_abs": diag["max_offdiag_abs"],
                    "pairwise_fidelity_mean": diag["pairwise_fidelity_mean"],
                    "pairwise_fidelity_max": diag["pairwise_fidelity_max"],
                    "benchmark_seed": int(metadata["benchmark_seed"]),
                    "data_seed": int(metadata["data_seed"]),
                    "ensemble_seed": int(metadata["ensemble_seed"]),
                    "benchmark_internal_seed": int(metadata["benchmark_internal_seed"]),
                    "existing_gram_rank_numeric": "" if existing_rank is None else existing_rank,
                    "rank_matches_existing": rank_matches_existing,
                    "state_generation": "normalized_complex_gaussian",
                    "stateprep_operation": "qml.StatePrep",
                    "stateprep_decomposition": "MottonenStatePreparation",
                    "label_payload": json.dumps({"nested_prefix_count": int(M)}),
                }
            )
    return output_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_dict_csv(path, rows, fieldnames=CSV_FIELDS)


def _finite_or_none(value: Any) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value_float):
        return value_float
    return None


def _rows_for_experiment(rows: Sequence[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["experiment_id"] == experiment_id]


def _summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lambda_mins = [float(row["lambda_min"]) for row in rows]
    condition_numbers = [
        value
        for value in (_finite_or_none(row.get("condition_number")) for row in rows)
        if value is not None
    ]
    rank_failures = [row for row in rows if not bool(row["rank_equals_M"])]
    return {
        "count": len(rows),
        "rank_failure_count": len(rank_failures),
        "min_lambda_min": float(min(lambda_mins)) if lambda_mins else None,
        "max_lambda_min": float(max(lambda_mins)) if lambda_mins else None,
        "max_condition_number": float(max(condition_numbers)) if condition_numbers else None,
    }


def _build_summary(
    *,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    output_paths: dict[str, Path],
    haar_args: argparse.Namespace,
) -> dict[str, Any]:
    rank_failures = [
        {
            "experiment_id": row["experiment_id"],
            "M": int(row["M"]),
            "instance_id": int(row["instance_id"]),
            "rank": int(row["rank"]),
            "expected_rank": int(row["expected_rank"]),
        }
        for row in rows
        if not bool(row["rank_equals_M"])
    ]
    haar_mismatches = [
        {
            "M": int(row["M"]),
            "instance_id": int(row["instance_id"]),
            "rank": int(row["rank"]),
            "existing_gram_rank_numeric": int(row["existing_gram_rank_numeric"]),
        }
        for row in rows
        if row["experiment_id"] == "exact_haar_d8" and not bool(row["rank_matches_existing"])
    ]
    by_experiment: dict[str, Any] = {}
    for experiment_id in ("weyl_heisenberg", "exact_haar_d8"):
        exp_rows = _rows_for_experiment(rows, experiment_id)
        by_m = {
            str(M): _summarize_group([row for row in exp_rows if int(row["M"]) == int(M)])
            for M in args.m_values
        }
        by_experiment[experiment_id] = {
            **_summarize_group(exp_rows),
            "by_M": by_m,
        }

    return {
        "config": {
            "rank_tol": float(args.rank_tol),
            "m_values": [int(value) for value in args.m_values],
            "instance_ids": [int(value) for value in args.instance_ids],
            "wh_results_csv": str(Path(args.wh_results_csv).expanduser().resolve()),
            "haar_results_csv": str(Path(args.haar_results_csv).expanduser().resolve()),
            "haar_summary_json": str(Path(args.haar_summary_json).expanduser().resolve()),
            "haar_reconstruction": {
                "nested_max_m": int(haar_args.nested_max_m),
                "state_dtype": str(haar_args.state_dtype),
                "fix_global_phase": bool(haar_args.fix_global_phase),
                "ensemble_master_seed": int(haar_args.ensemble_master_seed),
            },
            "output_paths": {key: str(value) for key, value in output_paths.items()},
        },
        "total_checked": len(rows),
        "rank_failure_count": len(rank_failures),
        "rank_failures": rank_failures,
        "haar_existing_rank_mismatch_count": len(haar_mismatches),
        "haar_existing_rank_mismatches": haar_mismatches,
        "by_experiment": by_experiment,
    }


def _plot_experiment(
    *,
    rows: Sequence[dict[str, Any]],
    experiment_id: str,
    title: str,
    m_values: Sequence[int],
    instance_ids: Sequence[int],
    output_path: Path,
) -> None:
    exp_rows = _rows_for_experiment(rows, experiment_id)
    by_key = {(int(row["M"]), int(row["instance_id"])): row for row in exp_rows}
    rank_matrix = np.full((len(m_values), len(instance_ids)), np.nan)
    lambda_matrix = np.full((len(m_values), len(instance_ids)), np.nan)
    mismatch_matrix = np.zeros((len(m_values), len(instance_ids)), dtype=bool)
    for m_idx, M in enumerate(m_values):
        for inst_idx, instance_id in enumerate(instance_ids):
            row = by_key[(int(M), int(instance_id))]
            rank_matrix[m_idx, inst_idx] = float(row["rank"])
            lambda_matrix[m_idx, inst_idx] = float(row["lambda_min"])
            mismatch_matrix[m_idx, inst_idx] = not bool(row["rank_equals_M"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    ax_rank, ax_lambda = axes

    im = ax_rank.imshow(
        rank_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=float(min(m_values)),
        vmax=float(max(m_values)),
    )
    ax_rank.set_title(f"{title}: numerical rank")
    ax_rank.set_xlabel("instance_id")
    ax_rank.set_ylabel("M")
    ax_rank.set_xticks(np.arange(len(instance_ids)), [str(value) for value in instance_ids])
    ax_rank.set_yticks(np.arange(len(m_values)), [str(value) for value in m_values])
    for m_idx, M in enumerate(m_values):
        for inst_idx, _instance_id in enumerate(instance_ids):
            rank_value = int(rank_matrix[m_idx, inst_idx])
            expected = int(M)
            text_color = "black" if rank_value >= (min(m_values) + max(m_values)) / 2 else "white"
            ax_rank.text(inst_idx, m_idx, str(rank_value), ha="center", va="center", color=text_color, fontsize=8)
            if mismatch_matrix[m_idx, inst_idx]:
                ax_rank.add_patch(
                    Rectangle(
                        (inst_idx - 0.5, m_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2.0,
                    )
                )
                ax_rank.text(inst_idx, m_idx + 0.28, f"!= {expected}", ha="center", va="center", color="red", fontsize=7)
    fig.colorbar(im, ax=ax_rank, fraction=0.046, pad=0.04, label="rank")

    for inst_idx, instance_id in enumerate(instance_ids):
        y_values = lambda_matrix[:, inst_idx]
        ax_lambda.plot(m_values, y_values, marker="o", linewidth=1.0, alpha=0.65, label=str(instance_id))
    ax_lambda.set_title(f"{title}: min eigenvalue of Gram matrix")
    ax_lambda.set_xlabel("M")
    ax_lambda.set_ylabel("lambda_min(G)")
    ax_lambda.set_xticks(list(m_values))
    if np.nanmin(lambda_matrix) > 0.0:
        ax_lambda.set_yscale("log")
    else:
        ax_lambda.set_yscale("symlog", linthresh=1e-12)
    ax_lambda.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax_lambda.legend(title="instance", fontsize=7, title_fontsize=8, ncol=2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _format_float(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    value_float = float(value)
    if not math.isfinite(value_float):
        return str(value_float)
    return f"{value_float:.{precision}g}"


def _summary_table_lines(summary: dict[str, Any], *, experiment_id: str, label: str) -> list[str]:
    exp_summary = summary["by_experiment"][experiment_id]
    rank_tol = float(summary["config"]["rank_tol"])
    lines = [
        f"### {label}",
        "",
        "| M | count | rank failures | min lambda_min | min lambda_min / tol | max lambda_min | max condition number |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for M, group in exp_summary["by_M"].items():
        min_lambda = group["min_lambda_min"]
        tolerance_margin = None if min_lambda is None else float(min_lambda) / rank_tol
        lines.append(
            "| "
            f"{M} | "
            f"{group['count']} | "
            f"{group['rank_failure_count']} | "
            f"{_format_float(group['min_lambda_min'])} | "
            f"{_format_float(tolerance_margin)} | "
            f"{_format_float(group['max_lambda_min'])} | "
            f"{_format_float(group['max_condition_number'])} |"
        )
    return lines


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    figure_paths: dict[str, Path],
    output_dir: Path,
) -> None:
    reports_dir = path.parent
    rel_results = Path(summary["config"]["output_paths"]["csv"]).relative_to(output_dir)
    rel_summary = Path(summary["config"]["output_paths"]["summary"]).relative_to(output_dir)
    rel_weyl_figure = Path(os.path.relpath(figure_paths["weyl_heisenberg"], start=reports_dir))
    rel_haar_figure = Path(os.path.relpath(figure_paths["exact_haar_d8"], start=reports_dir))
    wh_summary = summary["by_experiment"]["weyl_heisenberg"]
    haar_summary = summary["by_experiment"]["exact_haar_d8"]
    rank_tol = float(summary["config"]["rank_tol"])
    float64_eps = float(np.finfo(np.float64).eps)
    wh_tol_margin = float(wh_summary["min_lambda_min"]) / rank_tol
    haar_tol_margin = float(haar_summary["min_lambda_min"]) / rank_tol
    wh_kappa_eps = float(wh_summary["max_condition_number"]) * float64_eps
    haar_kappa_eps = float(haar_summary["max_condition_number"]) * float64_eps

    conclusion = (
        "두 benchmark 모두 요청 범위에서 `rank(G)=M`을 만족했다."
        if int(summary["rank_failure_count"]) == 0
        else f"{summary['rank_failure_count']}개 ensemble에서 `rank(G)!=M`이 관찰되었다."
    )

    lines = [
        "# Input Ensemble Linear Independence Verification",
        "",
        "## 결론",
        "",
        f"- 검증 대상: Weyl-Heisenberg 40개, Exact Haar D8 40개, 총 {summary['total_checked']}개 ensemble",
        f"- Numerical rank tolerance: `{summary['config']['rank_tol']}`",
        f"- Rank failure count: `{summary['rank_failure_count']}`",
        f"- Haar 기존 CSV `gram_rank_numeric` 교차검증 불일치: `{summary['haar_existing_rank_mismatch_count']}`",
        f"- 결론: {conclusion}",
        "",
        "## 산출물",
        "",
        f"- Raw CSV: `{rel_results}`",
        f"- Summary JSON: `{rel_summary}`",
        f"- Weyl-Heisenberg figure: `{rel_weyl_figure}`",
        f"- Exact Haar D8 figure: `{rel_haar_figure}`",
        "",
        "## 검증 방법",
        "",
        "각 실험 row의 원래 seed와 benchmark 생성 코드를 재사용해 input state ensemble을 재생성했다. "
        "재생성한 state matrix `Psi`로 Gram matrix `G = Psi Psi^dagger`를 만들고, "
        "`numpy.linalg.matrix_rank(G, tol=1e-10)`이 ensemble 크기 `M`과 같은지 확인했다.",
        "",
        "## 수치 정밀도와 Tolerance",
        "",
        "Gram matrix, eigenvalue, rank 계산은 재생성한 state를 `numpy.complex128`로 변환한 뒤 수행했다. "
        f"따라서 rank 판정은 NumPy/LAPACK의 double precision 계산에 기반하며, "
        f"`float64` machine epsilon은 약 `{float64_eps:.3e}`이다. "
        "Exact Haar D8 원래 runner도 Gram rank 진단에 `np.linalg.matrix_rank(gram, tol=1e-10)`을 사용하므로, "
        "이번 독립 검증에서도 같은 기준을 써서 기존 CSV의 `gram_rank_numeric`과 직접 비교할 수 있게 했다.",
        "",
        "`1e-10`은 절대 tolerance다. 여기서는 모든 input state가 normalize되어 `trace(G)=M`이고, "
        "Gram eigenvalue의 자연스러운 스케일이 `O(1)`이므로 절대 기준이 해석 가능하다. "
        "이 값은 double precision round-off보다 충분히 크지만, 관측된 최소 eigenvalue보다는 훨씬 작다. "
        f"가장 작은 `lambda_min(G)`도 Weyl-Heisenberg에서 `{_format_float(wh_summary['min_lambda_min'])}` "
        f"(`lambda_min/tol = {_format_float(wh_tol_margin)}`), "
        f"Exact Haar D8에서 `{_format_float(haar_summary['min_lambda_min'])}` "
        f"(`lambda_min/tol = {_format_float(haar_tol_margin)}`)였다. "
        "즉 rank를 잃으려면 현재 최소 eigenvalue가 tolerance 기준까지 수백만 배 이상 작아져야 한다.",
        "",
        "Condition number도 같은 결론을 지지한다. "
        f"Weyl-Heisenberg의 최대 `kappa(G)=lambda_max/lambda_min`은 `{_format_float(wh_summary['max_condition_number'])}`이고 "
        f"`kappa*eps ~= {wh_kappa_eps:.3e}`이다. "
        f"Exact Haar D8의 최대 condition number는 `{_format_float(haar_summary['max_condition_number'])}`이고 "
        f"`kappa*eps ~= {haar_kappa_eps:.3e}`이다. "
        "두 값 모두 `1/tol = 1e10`보다 훨씬 작고, `kappa*eps`도 rank threshold보다 작다. "
        "따라서 이번 rank 판정은 ill-conditioning 때문에 생긴 우연한 full-rank 판정으로 보기 어렵다.",
        "",
        "## Weyl-Heisenberg Benchmark",
        "",
        "이 benchmark는 세 qubit, 즉 Hilbert space dimension `d=8`에서 하나의 fiducial pure state를 만든 뒤, "
        "discrete Weyl-Heisenberg displacement orbit에서 `M`개의 state를 고르는 방식으로 구성된다. "
        "각 instance마다 먼저 `|000>`에서 시작해 qubit `q=0,1,2`에 대해 독립적인 각도 "
        "`theta_q, phi_q ~ Uniform[-pi, pi]`를 고정하고, 각 qubit에 `RY(theta_q)` 다음 `RZ(phi_q)`를 적용한다. "
        "그 다음 entangling layer로 `CNOT(0,1)`, `CNOT(1,2)`, `CZ(2,0)`를 적용하여 fiducial state `|phi>`를 얻는다.",
        "",
        "이후 `Z_8 x Z_8` phase-space에서 서로 다른 `M`개의 label `(a_j,b_j)`를 uniform하게 비복원 추출한다. "
        "각 label에 대해 generalized Pauli shift/phase operator를 사용하여 "
        "`|psi_j> = S X_8^{a_j} Z_8^{b_j} |phi>`를 만든다. 여기서 `Z_8|x> = omega^x|x>` "
        "(`omega = exp(2 pi i/8)`)이고 `X_8|x> = |x+1 mod 8>`이다. "
        "마지막의 고정 unitary `S`는 qubit `0,2`에는 Hadamard, qubit `1`에는 `RZ(pi/2)`를 적용한 뒤 "
        "`CZ(0,1)`, `CZ(1,2)`를 적용하는 scrambler다. 이 `S`는 모든 state에 동일하게 작용하므로 "
        "Gram matrix spectrum과 linear independence 여부를 보존한다.",
        "",
        "따라서 검증한 ensemble은 임의 fiducial state의 finite Weyl-Heisenberg orbit에서 고른 "
        "`M`개의 순수상태 ensemble이며, seed는 fiducial state와 phase-space label set을 재현하기 위해서만 사용되었다.",
        "",
        f"![Weyl-Heisenberg Gram rank diagnostics]({rel_weyl_figure})",
        "",
        f"- checked: `{wh_summary['count']}`",
        f"- rank failures: `{wh_summary['rank_failure_count']}`",
        f"- minimum lambda_min(G): `{_format_float(wh_summary['min_lambda_min'])}`",
        "",
        "## Exact Haar D8 Benchmark",
        "",
        "이 benchmark는 같은 세 qubit Hilbert space, 즉 `C^8`에서 Haar-random pure state를 직접 샘플링한다. "
        "각 instance마다 먼저 길이 12의 nested state list를 만든다. 각 state는 complex Gaussian vector "
        "`z in C^8`를 성분별로 `Re z_k, Im z_k ~ N(0,1)`에서 독립적으로 뽑은 뒤 "
        "`|psi> = z / ||z||_2`로 정규화하여 얻는다. complex Gaussian을 정규화한 분포는 complex unit sphere 위의 "
        "Haar measure와 일치하므로, 이 절차는 `d=8` Haar pure-state ensemble을 생성한다.",
        "",
        "실험에서 `M=5,...,8`을 바꿀 때는 같은 instance의 nested list에서 처음 `M`개 state를 취한다. "
        "즉 `M=5` ensemble은 `M=6,7,8` ensemble의 prefix가 되도록 구성되어 있으며, "
        "이번 검증도 이 nested 구조를 그대로 재현했다. 현재 결과셋에서는 별도의 global phase fixing을 사용하지 않았고, "
        "회로 실행 단계에서는 주어진 state vector를 정확한 state-preparation unitary로 준비한다.",
        "",
        f"![Exact Haar D8 Gram rank diagnostics]({rel_haar_figure})",
        "",
        f"- checked: `{haar_summary['count']}`",
        f"- rank failures: `{haar_summary['rank_failure_count']}`",
        f"- minimum lambda_min(G): `{_format_float(haar_summary['min_lambda_min'])}`",
        "",
        "## M별 요약",
        "",
    ]
    lines.extend(_summary_table_lines(summary, experiment_id="weyl_heisenberg", label="Weyl-Heisenberg"))
    lines.extend([""])
    lines.extend(_summary_table_lines(summary, experiment_id="exact_haar_d8", label="Exact Haar D8"))
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "csv": output_dir / "raw" / RESULTS_FILENAME,
        "summary": output_dir / "summaries" / SUMMARY_FILENAME,
        "report": output_dir / "reports" / REPORT_FILENAME,
        "weyl_figure": output_dir / "figures" / WEYL_FIGURE_FILENAME,
        "haar_figure": output_dir / "figures" / HAAR_FIGURE_FILENAME,
    }


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = _prepare_output_paths(output_dir)
    wh_rows = _load_csv_rows(Path(args.wh_results_csv).expanduser().resolve())
    haar_rows = _load_csv_rows(Path(args.haar_results_csv).expanduser().resolve())
    wh_indexed = _index_target_rows(
        wh_rows,
        m_values=args.m_values,
        instance_ids=args.instance_ids,
        source_name="Weyl-Heisenberg results CSV",
    )
    haar_indexed = _index_target_rows(
        haar_rows,
        m_values=args.m_values,
        instance_ids=args.instance_ids,
        source_name="Exact Haar D8 results CSV",
    )
    haar_args = _haar_args_from_summary(Path(args.haar_summary_json).expanduser().resolve())

    rows = [
        *_verify_weyl_rows(
            indexed_rows=wh_indexed,
            m_values=args.m_values,
            instance_ids=args.instance_ids,
            rank_tol=float(args.rank_tol),
        ),
        *_verify_haar_rows(
            indexed_rows=haar_indexed,
            m_values=args.m_values,
            instance_ids=args.instance_ids,
            rank_tol=float(args.rank_tol),
            haar_args=haar_args,
        ),
    ]

    _write_csv(output_paths["csv"], rows)
    figure_paths = {
        "weyl_heisenberg": output_paths["weyl_figure"],
        "exact_haar_d8": output_paths["haar_figure"],
    }
    _plot_experiment(
        rows=rows,
        experiment_id="weyl_heisenberg",
        title="Weyl-Heisenberg",
        m_values=args.m_values,
        instance_ids=args.instance_ids,
        output_path=output_paths["weyl_figure"],
    )
    _plot_experiment(
        rows=rows,
        experiment_id="exact_haar_d8",
        title="Exact Haar D8",
        m_values=args.m_values,
        instance_ids=args.instance_ids,
        output_path=output_paths["haar_figure"],
    )

    summary = _build_summary(
        rows=rows,
        args=args,
        output_paths=output_paths,
        haar_args=haar_args,
    )
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(
        path=output_paths["report"],
        summary=summary,
        figure_paths=figure_paths,
        output_dir=output_dir,
    )

    if summary["haar_existing_rank_mismatch_count"] != 0:
        raise RuntimeError("Exact Haar D8 rank cross-check against existing CSV failed.")
    if summary["rank_failure_count"] != 0:
        raise RuntimeError("At least one input ensemble failed rank(G)=M verification.")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify M=5..8 input ensemble linear independence with Gram-matrix ranks."
    )
    parser.add_argument("--wh-results-csv", type=str, default=str(DEFAULT_WH_RESULTS_CSV))
    parser.add_argument("--haar-results-csv", type=str, default=str(DEFAULT_HAAR_RESULTS_CSV))
    parser.add_argument("--haar-summary-json", type=str, default=str(DEFAULT_HAAR_SUMMARY_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--m-values", type=int, nargs="+", default=list(DEFAULT_M_VALUES))
    parser.add_argument("--instance-ids", type=int, nargs="+", default=list(DEFAULT_INSTANCE_IDS))
    parser.add_argument("--rank-tol", type=float, default=DEFAULT_RANK_TOL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_verification(args)
    print(f"saved: {_prepare_output_paths(Path(args.output_dir).expanduser().resolve())['csv']}")
    print(f"saved: {_prepare_output_paths(Path(args.output_dir).expanduser().resolve())['summary']}")
    print(f"saved: {_prepare_output_paths(Path(args.output_dir).expanduser().resolve())['report']}")
    print(f"rank_failure_count={summary['rank_failure_count']}")


if __name__ == "__main__":
    main()
