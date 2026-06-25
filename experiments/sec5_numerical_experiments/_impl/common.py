from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DEFAULT_RESULTS_CSV = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_sweep_results.csv"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_typed_csv_rows(
    paths: Sequence[str | Path],
    coerce_value: Callable[[str, str], Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        for row in _read_rows(Path(raw_path)):
            rows.append({key: coerce_value(key, value) for key, value in row.items()})
    return rows


def _load_jsonl_rows(paths: Sequence[str | Path], *, skip_missing: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if skip_missing and not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fieldnames_with_extras(rows: Sequence[dict[str, Any]], preferred: Sequence[str]) -> list[str]:
    preferred_list = [str(key) for key in preferred]
    extra = sorted(
        {
            str(key)
            for row in rows
            for key in row.keys()
            if str(key) not in preferred_list
        }
    )
    return preferred_list + extra


def _write_dict_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    fieldnames_list = [str(key) for key in fieldnames]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames_list})


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    summaries_dir = output_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "figures": figures_dir, "summaries": summaries_dir}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _quantile(values: Sequence[float], q: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    position = (len(sorted_values) - 1) * float(q)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = position - lo
    return float(sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    payload = {
        "count": int(len(values)),
        "median": float(statistics.median(values)),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }
    if len(values) > 1:
        payload["mean"] = float(statistics.fmean(values))
        payload["pstdev"] = float(statistics.pstdev(values))
    else:
        payload["mean"] = float(values[0])
        payload["pstdev"] = 0.0
    return payload


def _all_rows(series_rows: dict[str, Sequence[dict[str, str]]]) -> list[dict[str, str]]:
    return [row for rows in series_rows.values() for row in rows]


def _aggregate_by_m(
    series_rows: dict[str, Sequence[dict[str, str]]],
    model_configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    m_values = sorted({_int(row, "M") for row in _all_rows(series_rows)})
    for m_value in m_values:
        first_bucket = [
            row
            for rows in series_rows.values()
            for row in rows
            if _int(row, "M") == m_value
        ]
        first = first_bucket[0]
        item: dict[str, Any] = {
            "n_sys": _int(first, "n_sys"),
            "d": _int(first, "d"),
            "M": int(m_value),
            "M_over_d": _float(first, "M_over_d"),
        }
        for model_key, config in model_configs.items():
            rows = series_rows.get(model_key, [])
            bucket = [row for row in rows if _int(row, "M") == m_value]
            if not bucket:
                continue
            values = [_float(row, str(config["column"])) for row in bucket]
            item[model_key] = _stats(values)
        payload.append(item)
    return payload


def _with_line_gap(
    x_values: Sequence[float],
    y_values: Sequence[float],
    m_values: Sequence[int],
    *,
    gap_after_m: int = 8,
) -> tuple[list[float], list[float]]:
    x_with_gap: list[float] = []
    y_with_gap: list[float] = []
    for m_value, x_value, y_value in zip(m_values, x_values, y_values, strict=True):
        x_with_gap.append(float(x_value))
        y_with_gap.append(float(y_value))
        if int(m_value) == int(gap_after_m):
            x_with_gap.append(float("nan"))
            y_with_gap.append(float("nan"))
    return x_with_gap, y_with_gap
