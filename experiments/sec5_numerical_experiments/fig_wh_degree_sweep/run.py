from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl" / "wh_degree_sweep.py"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "sec5_numerical_experiments"
    / "fig_wh_degree_sweep"
    / "results"
    / "wh_md_walsh_degree_sweep_nsys3_M9_M12_drop_extra_i5_r15"
)
DEFAULT_REFERENCE = (
    ROOT
    / "data"
    / "paper"
    / "fig_wh_d8_sweep"
    / "raw"
    / "wh_md_sweep_results.csv"
)


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run WH Walsh-degree sweep for fig:wh-sweep.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-results-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--jax-platform",
        choices=["gpu", "cpu"],
        default="gpu",
        help="JAX platform for paper reproductions. Use cpu for local mechanics checks.",
    )
    known, passthrough = parser.parse_known_args(argv)

    args = list(passthrough)
    args = _with_default(args, "--output-dir", (str(known.output_dir.expanduser().resolve()),))
    args = _with_default(
        args,
        "--reference-results-csv",
        (str(known.reference_results_csv.expanduser().resolve()),),
    )
    args = _with_default(args, "--jit-backend", (str(known.jax_platform),))
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = str(known.jax_platform)
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True, env=env)


if __name__ == "__main__":
    main()
