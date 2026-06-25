from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
IMPL_DIR = ROOT / "experiments" / "sec5_numerical_experiments" / "_impl"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "sec5_numerical_experiments" / "fig_wh_d8_sweep" / "results"
PAPER_M_VALUES = ("5", "6", "7", "8", "9", "10", "11", "12")
PAPER_INSTANCE_IDS = tuple(str(value) for value in range(10))


def _has_arg(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(arg.startswith(f"{flag}=") for arg in args)


def _with_default(args: list[str], flag: str, values: Sequence[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def _paper_wh_args(args: list[str], *, jit_backend_default: str) -> list[str]:
    defaults = (
        ("--n-sys-list", ("3",)),
        ("--m-values", PAPER_M_VALUES),
        ("--instance-ids", PAPER_INSTANCE_IDS),
        ("--num-restarts", ("50",)),
        ("--steps", ("1000",)),
        ("--eval-interval", ("50",)),
        ("--su-depth", ("14",)),
        ("--scale-init", ("1.0",)),
        ("--bias-scale-init", ("1.0",)),
        ("--projection-strategy", ("drop_extra",)),
        ("--jit-backend", (jit_backend_default,)),
    )
    out = list(args)
    for flag, values in defaults:
        out = _with_default(out, flag, values)
    return out


def _random_sparse_args(args: list[str]) -> list[str]:
    value_options = {
        "--m-values",
        "--instance-ids",
        "--reference-summary-json",
        "--sparse-seed-offset",
        "--random-sparse-execution",
        "--device-name",
        "--diff-method",
        "--jit-backend",
        "--state-dtype",
        "--plot-dpi",
    }
    flag_options = {"--aggregate-only"}

    out: list[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token in flag_options:
            out.append(token)
            idx += 1
            continue
        if token in value_options:
            out.append(token)
            idx += 1
            while idx < len(args) and not args[idx].startswith("--"):
                out.append(args[idx])
                idx += 1
            continue
        if any(token.startswith(f"{flag}=") for flag in value_options | flag_options):
            out.append(token)
        idx += 1
    return out


def _run(script: Path, args: list[str], *, jax_platform: str) -> None:
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = str(jax_platform)
    subprocess.run([sys.executable, str(script), *args], check=True, env=env)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Weyl-Heisenberg d=8 experiments for fig:wh."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["full_ucr", "walsh_degree_1", "random_sparse_ucr"],
        default=["full_ucr", "walsh_degree_1", "random_sparse_ucr"],
        help=(
            "Model groups to run for Fig. fig:wh. WD-1 uses the Walsh parity runner; "
            "random_sparse_ucr uses the full_ucr reference CSV."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--jax-platform",
        choices=["gpu", "cpu"],
        default="gpu",
        help="JAX platform for paper reproductions. Use cpu for local mechanics checks.",
    )
    known, passthrough = parser.parse_known_args(argv)

    output_root = known.output_root.expanduser().resolve()
    selected = set(known.models)
    jax_platform = str(known.jax_platform)

    if "full_ucr" in selected:
        args = _paper_wh_args(list(passthrough), jit_backend_default=jax_platform)
        args = _with_default(args, "--output-dir", (str(output_root / "full_ucr"),))
        _run(IMPL_DIR / "wh_d8_sweep.py", args, jax_platform=jax_platform)

    if "walsh_degree_1" in selected:
        args = _paper_wh_args(list(passthrough), jit_backend_default=jax_platform)
        args = _with_default(args, "--output-dir", (str(output_root / "walsh_degree1"),))
        args = _with_default(args, "--jax-platform", (jax_platform,))
        _run(
            ROOT / "experiments" / "sec5_numerical_experiments" / "fig_wh_d8_sweep" / "run_wd1.py",
            args,
            jax_platform=jax_platform,
        )

    if "random_sparse_ucr" in selected:
        args = _random_sparse_args(list(passthrough))
        args = _with_default(args, "--m-values", PAPER_M_VALUES)
        args = _with_default(args, "--instance-ids", PAPER_INSTANCE_IDS)
        args = _with_default(args, "--jit-backend", (jax_platform,))
        args = _with_default(args, "--output-dir", (str(output_root / "random_sparse_ucr"),))
        args = _with_default(
            args,
            "--reference-results-csv",
            (str(output_root / "full_ucr" / "raw" / "wh_md_sweep_results.csv"),),
        )
        _run(
            IMPL_DIR / "random_sparse_model" / "random_sparse_ucr_vs_degree1.py",
            args,
            jax_platform=jax_platform,
        )


if __name__ == "__main__":
    main()
