from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl" / "haar_d8" / "haar_d8_exact_sweep_restart_reuse.py"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "sec5_numerical_experiments"
    / "fig_haar_d8_sweep"
    / "results"
    / "exact_haar_d8_i10_r50"
)


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run exact Haar d=8 sweep for fig:haar.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jax-platform",
        choices=["gpu", "cpu"],
        default="gpu",
        help="JAX platform for paper reproductions. Use cpu for local mechanics checks.",
    )
    known, passthrough = parser.parse_known_args(argv)

    args = list(passthrough)
    args = _with_default(args, "--output-dir", (str(known.output_dir.expanduser().resolve()),))
    args = _with_default(args, "--jit-backend", (str(known.jax_platform),))
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = str(known.jax_platform)
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True, env=env)


if __name__ == "__main__":
    main()
