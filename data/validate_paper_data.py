from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "paper_data_manifest.json"


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
        return {"format": "jsonl", "record_count": sum(1 for line in handle if line.strip())}


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


def _compare_schema(
    *,
    path: Path,
    expected: dict[str, Any],
    actual: dict[str, Any],
    allow_row_count_drift: bool,
) -> list[str]:
    errors: list[str] = []
    if actual.get("format") != expected.get("format"):
        errors.append(f"{path}: format mismatch {actual.get('format')!r} != {expected.get('format')!r}")
        return errors
    if expected.get("format") == "csv":
        if actual.get("columns") != expected.get("columns"):
            errors.append(f"{path}: CSV columns differ")
        if not allow_row_count_drift and actual.get("row_count") != expected.get("row_count"):
            errors.append(f"{path}: row_count {actual.get('row_count')} != {expected.get('row_count')}")
    elif expected.get("format") == "jsonl" and not allow_row_count_drift:
        if actual.get("record_count") != expected.get("record_count"):
            errors.append(f"{path}: record_count {actual.get('record_count')} != {expected.get('record_count')}")
    elif expected.get("format") == "json":
        if actual.get("top_level_type") != expected.get("top_level_type"):
            errors.append(f"{path}: JSON top-level type differs")
        if actual.get("top_level_keys") != expected.get("top_level_keys"):
            errors.append(f"{path}: JSON top-level keys differ")
    return errors


def validate(
    *,
    manifest_path: Path,
    schema_only: bool,
    allow_row_count_drift: bool,
) -> tuple[int, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = 0
    for record in manifest.get("files", []):
        path = ROOT / record["path"]
        checked += 1
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        if not schema_only:
            actual_hash = _sha256(path)
            if actual_hash != record.get("sha256"):
                errors.append(f"{path}: sha256 {actual_hash} != {record.get('sha256')}")
        errors.extend(
            _compare_schema(
                path=path,
                expected=record.get("schema", {}),
                actual=_file_schema(path),
                allow_row_count_drift=allow_row_count_drift,
            )
        )
    return checked, errors


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate paper data against the manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Ignore checksums but require the same columns/top-level JSON shape and counts.",
    )
    parser.add_argument(
        "--allow-row-count-drift",
        action="store_true",
        help="With --schema-only, allow reduced-budget outputs with fewer/more rows.",
    )
    args = parser.parse_args(argv)

    checked, errors = validate(
        manifest_path=args.manifest.expanduser().resolve(),
        schema_only=args.schema_only,
        allow_row_count_drift=args.allow_row_count_drift,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    mode = "schema" if args.schema_only else "exact"
    print(f"validated_files={checked}")
    print(f"mode={mode}")


if __name__ == "__main__":
    main()
