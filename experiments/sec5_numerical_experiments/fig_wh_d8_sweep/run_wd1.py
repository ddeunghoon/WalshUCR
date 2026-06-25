from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
IMPL_SCRIPT = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl" / "wh_d8_walsh_degree1_sweep.py"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "sec5_numerical_experiments"
    / "fig_wh_d8_sweep"
    / "results"
    / "wh_md_walsh_degree1_nsys3_scale1_drop_extra_i10_r50"
)


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jax-platform", choices=["gpu", "cpu"], default="gpu")
    known, passthrough = parser.parse_known_args(argv)

    args = list(passthrough)
    os.environ["JAX_PLATFORM_NAME"] = str(known.jax_platform)

    if "-h" not in args and "--help" not in args:
        args = _with_default(args, "--output-dir", (str(DEFAULT_OUTPUT),))
        args = _with_default(args, "--jit-backend", (str(known.jax_platform),))

    sys.argv = [str(IMPL_SCRIPT), *args]
    runpy.run_path(str(IMPL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
