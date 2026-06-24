# WalshUCR Reproducibility Package

This repository provides source code for reproducing the Weyl--Heisenberg
Walsh-degree experiments associated with the manuscript:

> Walsh-structured uniformly controlled rotations for variational quantum state
> discrimination

The package is organized so that readers and reviewers can check the software
environment, run a small end-to-end calculation, and regenerate the dense
Walsh-degree-1 sweep outputs used by the manuscript's `d=8` Weyl--Heisenberg
comparison.

## Reproducibility Scope

This artifact supports:

- construction of Walsh-truncated VQSD model classes;
- deterministic Weyl--Heisenberg ensemble generation from recorded seeds;
- SDP reference computation through CVXPY;
- restart-based optimization for the Walsh-degree-1 model;
- aggregation of per-instance outputs into CSV, JSON summary, and a gap plot;
- release sanity checks through `pytest`.

This artifact does not contain generated result directories, checkpoint files,
profiler traces, local virtual environments, or large archive files. Those files
are intentionally excluded so that the repository remains source-only.

The scripts in this repository are the compact CPU/PennyLane reproduction path
for the Weyl--Heisenberg Walsh-degree experiment. Larger GPU-only checks and any
additional archived numerical outputs should be cited separately if they are
needed to reproduce other manuscript tables or figures.

## Repository Layout

```text
src/scalable_vqsd/
  Model, benchmark, trainer, and utility code.

experiments/ucr_method/sec5/
  Walsh-degree-1 sweep, restart-reuse runner, and parallel aggregation wrapper.

tests/
  Lightweight tests for package import, deterministic benchmark data, model
  parameterization, and CLI availability.
```

Key entry points:

- `experiments/ucr_method/sec5/wh_md_walsh_degree1_sweep.py`
- `experiments/ucr_method/sec5/run_wh_md_walsh_degree1_nsys3_scale1_i10_r50_restart_reuse_parallel.sh`

## Environment

Required:

- Python 3.11
- `uv`

Optional:

- GNU Parallel for the sharded dense sweep wrapper.

The environment is pinned by `uv.lock`. All Python commands below are written
with `uv` so that they run in the locked environment.

The release scripts default to CPU execution. A CPU-only machine is sufficient
for the checks below. If a CUDA-capable GPU is visible but the installed JAX
wheel is CPU-only, JAX may print a fallback warning; this does not affect the
CPU reproducibility checks.

## Install

From the repository root:

```bash
uv sync --frozen
```

## Level 1: Environment and Import Checks

These commands verify that the lockfile resolves, Python files compile, and the
smoke tests pass:

```bash
uv lock --check
uv sync --frozen
uv run python -m compileall -q src experiments tests
uv run pytest
```

Expected result for the test suite:

```text
4 passed
```

## Level 2: Minimal End-to-End Run

The following command runs one tiny Weyl--Heisenberg instance with one restart
and one optimizer step. It is intended to validate the full execution path,
including problem generation, SDP reference computation, optimization,
checkpoint writing, aggregation, and plotting.

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

Expected output files:

```text
/tmp/walshucr_smoke/raw/wh_md_walsh_degree1_results.csv
/tmp/walshucr_smoke/raw/wh_md_walsh_degree1_restart_records.jsonl
/tmp/walshucr_smoke/raw/restart_checkpoints/nsys1_M2_instance00_walsh_degree_1.jsonl
/tmp/walshucr_smoke/figures/wh_md_walsh_degree1_gap_left_panel.png
/tmp/walshucr_smoke/summaries/wh_md_walsh_degree1_summary.json
```

The exact optimized value is not a manuscript result for this tiny run; the run
only checks that all code paths complete and materialize the expected artifacts.

## Level 3: Dense `d=8` Walsh-Degree-1 Sweep

The main script defaults to the manuscript-scale Weyl--Heisenberg
Walsh-degree-1 grid:

- `n_sys = 3` (`d = 8`)
- `M = 5, 6, 7, 8, 9, 10, 11, 12`
- 10 ensemble instances for each `M`
- 50 optimization restarts for each instance
- 1000 optimization steps per restart
- uniform priors and the `drop_extra` projection rule

Single-process execution:

```bash
JAX_PLATFORM_NAME=cpu uv run python experiments/ucr_method/sec5/wh_md_walsh_degree1_sweep.py
```

This is a long-running nonconvex optimization workload. The parallel wrapper is
the recommended path for regenerating the dense sweep because it shards over
`M` and instance id, reuses completed shard outputs, and performs a final
aggregation pass:

```bash
bash experiments/ucr_method/sec5/run_wh_md_walsh_degree1_nsys3_scale1_i10_r50_restart_reuse_parallel.sh
```

Default output location:

```text
experiments/ucr_method/sec5/results/wh_md_walsh_degree1_nsys3_scale1_drop_extra_restart_reuse_i10_r50/
```

Final aggregate outputs are written under:

```text
.../final/raw/wh_md_walsh_degree1_results.csv
.../final/raw/wh_md_walsh_degree1_restart_records.jsonl
.../final/figures/wh_md_walsh_degree1_gap_left_panel.png
.../final/summaries/wh_md_walsh_degree1_summary.json
```

Generated `results/` directories are ignored by git.

## Reduced-Budget Reviewer Run

To inspect the dense-sweep workflow without running the full optimization
budget, override the grid and restart count:

```bash
M_LIST="5 6" INSTANCE_IDS="0 1" NUM_RESTARTS=2 STEPS=50 \
bash experiments/ucr_method/sec5/run_wh_md_walsh_degree1_nsys3_scale1_i10_r50_restart_reuse_parallel.sh
```

This reduced-budget run is useful for reviewing the mechanics of sharding,
checkpointing, aggregation, and output formats. It is not expected to reproduce
the manuscript numerical values.

## Output Interpretation

The main CSV contains one row per ensemble instance. Important fields include:

- `n_sys`, `d`, `M`, and `instance_id`;
- deterministic `benchmark_seed` and `data_seed`;
- `p_opt`, the SDP reference success probability;
- `p_succ_walsh_deg1`, the best terminal success probability over restarts;
- `gap_abs_walsh_deg1 = p_opt - p_succ_walsh_deg1`;
- restart metadata such as `best_restart_walsh_deg1`, `seed_opt_walsh_deg1`,
  `num_steps_walsh_deg1`, and `termination_reason_walsh_deg1`.

The summary JSON records the run configuration, aggregate statistics grouped by
`(n_sys, M)`, and paths to generated artifacts.

Because the optimization is nonconvex, the manuscript-scale settings use a fixed
restart protocol. Small numerical differences can occur across platforms,
linear-algebra libraries, or JAX/PennyLane execution details; the seeds and
optimization budget are fixed to make such comparisons auditable.

## Citation and License

Citation metadata is provided in `CITATION.cff`. The code is released under the
MIT License; see `LICENSE`.
