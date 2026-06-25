from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "experiments"
    / "sec5_numerical_experiments"
    / "table_d16_checks"
    / "gpu"
    / "run_walsh_degree1_gpu_memopt.py"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "sec5_numerical_experiments"
    / "table_d16_checks"
    / "results"
    / "single_gpu_probe"
)


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a d=16 GPU check instance.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jax-platform",
        choices=["gpu", "cpu"],
        default="gpu",
        help="JAX platform. CPU is intended only for small mechanics checks.",
    )
    known, passthrough = parser.parse_known_args(argv)

    args = list(passthrough)
    for flag, values in (
        ("--n-sys", ("4",)),
        ("--M", ("16",)),
        ("--instance-id", ("0",)),
        ("--model-type", ("walsh_degree_1",)),
        ("--state-family", ("weyl",)),
        ("--num-restarts", ("5",)),
        ("--su-depth", ("61",)),
        ("--steps", ("1000",)),
        ("--eval-interval", ("50",)),
        ("--scale-init", ("1.0",)),
        ("--bias-scale-init", ("1.0",)),
        ("--output-dir", (str(known.output_dir.expanduser().resolve()),)),
    ):
        args = _with_default(args, flag, values)

    if str(known.jax_platform) == "cpu":
        args = _with_default(args, "--no-require-gpu", ())

    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = str(known.jax_platform)
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True, env=env)


if __name__ == "__main__":
    main()
