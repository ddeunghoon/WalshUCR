# WalshUCR

Source-code release for the Walsh-structured uniformly controlled rotation (UCR)
experiments accompanying the manuscript:

> Walsh-structured uniformly controlled rotations for variational quantum state
> discrimination

This repository is intended as the compact reproduction package for the
Weyl--Heisenberg Walsh-degree experiments. It contains the Python package used by
the release scripts, the Section 5 Walsh-degree-1 sweep driver, the restart-reuse
runner, and a GNU Parallel wrapper for the dense `d=8` sweep.

## Scope

Included:

- Walsh-truncated VQSD model classes and the recursive CSD-style ansatz code.
- Weyl--Heisenberg benchmark generation with deterministic seeds.
- SDP reference computation through CVXPY.
- The Walsh degree-1 Section 5 sweep script and aggregation path.
- A small pytest smoke test suite.

Not included:

- Large generated result directories, restart checkpoints, figures, or profiler
  output.
- Local virtual environments.
- The separate GPU-oriented experiment tree used for larger `d=16` checks.

If this repository is cited as the code availability artifact for the full
manuscript, state clearly which manuscript figures/tables are reproduced by this
release and where any additional GPU result code or archived data are hosted.

## Requirements

- Python 3.11
- `uv`
- Optional for the parallel sweep: GNU Parallel

The lockfile pins the release environment. By default the included scripts use
the CPU backend (`JAX_PLATFORM_NAME=cpu`). Installing a CUDA-enabled JAX stack is
environment-specific and is not required for the smoke tests below.

## Setup

From this directory:

```bash
uv sync --frozen
```

Run all Python commands through `uv`, for example:

```bash
uv run python -m pytest
```

## Quick Checks

These checks should complete quickly on a CPU-only machine:

```bash
uv lock --check
uv sync --frozen
uv run python -m compileall -q src experiments tests
uv run pytest
```

A minimal end-to-end run that writes temporary artifacts:

```bash
rm -rf /tmp/walshucr_smoke
JAX_PLATFORM_NAME=cpu uv run python experiments/ucr_method/sec5/wh_md_walsh_degree1_sweep.py \
  --n-sys-list 1 \
  --m-values 2 \
  --instance-ids 0 \
  --num-instances-per-grid-point 1 \
  --num-restarts 1 \
  --steps 1 \
  --eval-interval 1 \
  --su-depth 1 \
  --output-dir /tmp/walshucr_smoke
```

Expected outputs:

- `/tmp/walshucr_smoke/raw/wh_md_walsh_degree1_results.csv`
- `/tmp/walshucr_smoke/raw/wh_md_walsh_degree1_restart_records.jsonl`
- `/tmp/walshucr_smoke/figures/wh_md_walsh_degree1_gap_left_panel.png`
- `/tmp/walshucr_smoke/summaries/wh_md_walsh_degree1_summary.json`

## Reproducing the Dense Walsh-Degree-1 Sweep

Single-process run using the defaults encoded in the script:

```bash
JAX_PLATFORM_NAME=cpu uv run python experiments/ucr_method/sec5/wh_md_walsh_degree1_sweep.py
```

The defaults are the `d=8` Weyl--Heisenberg grid:

- `n_sys = 3`
- `M = 5, 6, 7, 8, 9, 10, 11, 12`
- 10 instances per `M`
- 50 optimization restarts per instance
- 1000 optimization steps

This is a long-running nonconvex optimization workload. For the parallel version
with shard reuse and final aggregation:

```bash
bash experiments/ucr_method/sec5/run_wh_md_walsh_degree1_nsys3_scale1_i10_r50_restart_reuse_parallel.sh
```

The wrapper uses GNU Parallel and writes outputs under
`experiments/ucr_method/sec5/results/` by default. The `results/` paths are
ignored by git so regenerated artifacts are not accidentally committed.

You can override the grid or budget with environment variables:

```bash
M_LIST="5 6" INSTANCE_IDS="0 1" NUM_RESTARTS=2 STEPS=50 \
bash experiments/ucr_method/sec5/run_wh_md_walsh_degree1_nsys3_scale1_i10_r50_restart_reuse_parallel.sh
```

## Repository Hygiene

For GitHub release, add the directory with:

```bash
git add WalshUCR/
```

Do not add local archives or environments such as `Walsh.tar.gz` or
`WalshUCR/.venv/`. The repository-level `.gitignore` excludes local environments,
Python caches, build outputs, logs, result directories, and generated figures.

## License

This code is released under the MIT License. See `LICENSE`.

## Citation

Please cite the associated manuscript and this repository. A machine-readable
template is provided in `CITATION.cff`.
