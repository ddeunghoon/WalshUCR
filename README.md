# WalshUCR Reproducibility Artifact

This repository supports the paper:

> Walsh-structured uniformly controlled rotations for variational quantum state
> discrimination

WalshUCR is organized for readers and reviewers who want to inspect the code,
validate the included numerical data, rebuild the paper figures, or rerun the
experiments. Generated `results/` directories are ignored by git. Large restart
logs, checkpoint payloads, profiler traces, and full GPU raw files are not
required to rebuild the reported figures and tables and should be archived
outside this repository if retained.

## Repository Layout

```text
src/walsh_ucr/
  Model, benchmark, trainer, and utility code.

experiments/
  Paper experiment entry points grouped by section.

experiments/sec5_numerical_experiments/fig_wh_d8_sweep/
  Weyl--Heisenberg d=8 full-UCR, WD-1, and RS-UCR runners.

experiments/sec5_numerical_experiments/fig_haar_d8_sweep/
  Exact-Haar d=8 full-UCR, WD-1, and RS-UCR runner.

experiments/sec5_numerical_experiments/fig_wh_degree_sweep/
  Weyl--Heisenberg Walsh-degree sweep runner.

experiments/sec5_numerical_experiments/table_d16_checks/
  GPU-oriented d=16 Weyl/Haar runner, batch scripts, and table aggregator.

experiments/appendix/rank_diagnostics/
  Gram-rank diagnostic runner.

data/paper/
  Compact data files used to rebuild paper figures and tables.

data/manifests/
  Machine-readable inventories and checksums.

figures/
  Data-driven paper figure generation.

tests/
  Lightweight import, model, benchmark, and CLI checks.
```

`data/paper/` means the data included for reproducing the paper outputs. It is
not a Python package data directory and is not a dump of all exploratory output.

## Paper Output Map

| Paper output | Description | Code | Data |
|---|---|---|---|
| Table `tab:param-decomposition` | `d=8` ansatz parameter counts | `data/prepare_paper_data.py` | `data/paper/table_param_decomposition/` |
| Fig. `fig:wh` | Weyl--Heisenberg `d=8` M-sweep | `experiments/sec5_numerical_experiments/fig_wh_d8_sweep/` | `data/paper/fig_wh_d8_sweep/` |
| Fig. `fig:haar` | Haar-random `d=8` M-sweep | `experiments/sec5_numerical_experiments/fig_haar_d8_sweep/` | `data/paper/fig_haar_d8_sweep/` |
| Fig. `fig:wh-sweep` | WH overcomplete Walsh-degree sweep | `experiments/sec5_numerical_experiments/fig_wh_degree_sweep/` | `data/paper/fig_wh_degree_sweep/` |
| Table `tab:d16-checks` | Representative `d=16` checks | `experiments/sec5_numerical_experiments/table_d16_checks/` | `data/paper/table_d16_checks/` |
| Table `tab:app-ensemble-grid` | Dimensions, M values, and instance counts | `data/prepare_paper_data.py` | `data/paper/table_app_ensemble_grid/` |
| Table `tab:app-rank-diagnostics` | Gram-rank diagnostics | `experiments/appendix/rank_diagnostics/` | `data/paper/appendix_rank_diagnostics/` |

## Environment

Required:

- Python 3.11
- `uv`

Optional:

- CUDA-enabled JAX for paper-budget numerical reproduction. CPU execution
  remains available for small mechanics checks.

Install from the repository root:

```bash
uv sync --frozen
```

All Python commands below use `uv` and the locked environment.

## Quick Checks

```bash
uv lock --check
uv sync --frozen
uv run python -m compileall -q src experiments data figures tests
uv run pytest
```

Expected test-suite result:

```text
6 passed
```

## Data and Figures

Validate the included paper data exactly:

```bash
uv run python data/validate_paper_data.py
```

Check only that regenerated outputs have the same schema:

```bash
uv run python data/validate_paper_data.py --schema-only
```

Regenerate the numerical paper PDF/PNG figures from `data/paper`:

```bash
uv run python figures/build_paper_figures.py
```

Outputs are written under:

```text
figures/paper/
```

The data checksum and schema manifest is:

```text
data/manifests/paper_data_manifest.json
```

The paper artifact inventory is:

```text
data/manifests/paper_results_manifest.toml
```

## Preparing Included Data

The committed data can be rebuilt from the local canonical result roots in the
parent project:

```bash
uv run python data/prepare_paper_data.py --skip-d16
```

Omit `--skip-d16` only when the six full d=16 GPU result roots listed in
`data/prepare_paper_data.py` are available locally.

## Minimal End-to-End Run

This tiny CPU run checks problem generation, SDP reference computation,
optimization, checkpoint writing, aggregation, and plotting. It is not a paper
result.

```bash
rm -rf /tmp/walshucr_fig_wh_smoke
uv run python experiments/sec5_numerical_experiments/fig_wh_d8_sweep/run_wd1.py \
  --jax-platform cpu \
  --n-sys-list 1 \
  --m-values 2 \
  --instance-ids 0 \
  --num-instances-per-grid-point 1 \
  --num-restarts 1 \
  --steps 1 \
  --eval-interval 1 \
  --su-depth 1 \
  --output-dir /tmp/walshucr_fig_wh_smoke
```

Expected output files include:

```text
/tmp/walshucr_fig_wh_smoke/raw/wh_md_walsh_degree1_results.csv
/tmp/walshucr_fig_wh_smoke/raw/wh_md_walsh_degree1_restart_records.jsonl
/tmp/walshucr_fig_wh_smoke/figures/wh_md_walsh_degree1_gap_left_panel.png
/tmp/walshucr_fig_wh_smoke/summaries/wh_md_walsh_degree1_summary.json
```

## Paper-Budget Runs

The WH `d=8` runner covers full-UCR, WD-1, and RS-UCR paths for Fig. `fig:wh`:

```bash
uv run python experiments/sec5_numerical_experiments/fig_wh_d8_sweep/run.py
```

Default configuration:

- `n_sys = 3`, so `d = 8`
- `M = 5, 6, 7, 8, 9, 10, 11, 12`
- 10 ensemble instances for each `M`
- 50 optimization restarts for each instance
- 1000 optimization steps per restart
- uniform priors and the `drop_extra` projection rule

Other entry points:

```bash
# Exact Haar d=8
uv run python experiments/sec5_numerical_experiments/fig_haar_d8_sweep/run.py

# WH Walsh-degree sweep
uv run python experiments/sec5_numerical_experiments/fig_wh_degree_sweep/run.py

# Appendix rank diagnostics
uv run python experiments/appendix/rank_diagnostics/run.py

# One d=16 GPU-runner probe
uv run python experiments/sec5_numerical_experiments/table_d16_checks/run_gpu.py --skip-sdp
```

The `d=16` paper-budget runs require a CUDA-enabled JAX installation on the
target machine. Batch scripts are under
`experiments/sec5_numerical_experiments/table_d16_checks/gpu/`.

Compact table-ready d=16 data can be regenerated from the six full result roots
with:

```bash
uv run python experiments/sec5_numerical_experiments/table_d16_checks/aggregate_table.py \
  --input-roots <six d16 result roots> \
  --output-dir data/paper/table_d16_checks
```

## Reduced-Budget Reviewer Run

```bash
uv run python experiments/sec5_numerical_experiments/fig_wh_d8_sweep/run.py \
  --models walsh_degree_1 \
  --m-values 5 6 \
  --instance-ids 0 1 \
  --num-restarts 2 \
  --steps 50
```

This reduced run is for checking checkpoint reuse, aggregation, and output
formats. It is not expected to reproduce paper numerical values.

## License

This repository is released under the MIT License. See `LICENSE`.
