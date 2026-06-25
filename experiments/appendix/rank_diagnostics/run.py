from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "appendix" / "rank_diagnostics" / "verify_input_ensemble_linear_independence.py"
DEFAULT_WH_RESULTS = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_walsh_degree1_results.csv"
)
DEFAULT_HAAR_RESULTS = (
    ROOT
    / "data"
    / "paper"
    / "fig_haar_d8_sweep"
    / "raw"
    / "exact_haar_d8_sweep_results.csv"
)
DEFAULT_HAAR_SUMMARY = (
    ROOT
    / "data"
    / "paper"
    / "fig_haar_d8_sweep"
    / "summaries"
    / "exact_haar_d8_sweep_summary.json"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "appendix" / "rank_diagnostics" / "results"


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run appendix input-ensemble Gram-rank diagnostics.")
    parser.add_argument("--wh-results-csv", type=Path, default=DEFAULT_WH_RESULTS)
    parser.add_argument("--haar-results-csv", type=Path, default=DEFAULT_HAAR_RESULTS)
    parser.add_argument("--haar-summary-json", type=Path, default=DEFAULT_HAAR_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    known, passthrough = parser.parse_known_args(argv)

    args = list(passthrough)
    args = _with_default(args, "--wh-results-csv", (str(known.wh_results_csv.expanduser().resolve()),))
    args = _with_default(args, "--haar-results-csv", (str(known.haar_results_csv.expanduser().resolve()),))
    args = _with_default(args, "--haar-summary-json", (str(known.haar_summary_json.expanduser().resolve()),))
    args = _with_default(args, "--output-dir", (str(known.output_dir.expanduser().resolve()),))
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True)


if __name__ == "__main__":
    main()
