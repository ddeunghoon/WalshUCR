# Code Availability Scope

This repository is a source-only reproducibility artifact for the
Weyl--Heisenberg Walsh-degree experiments associated with the manuscript
"Walsh-structured uniformly controlled rotations for variational quantum state
discrimination."

The artifact is designed for readers and reviewers who want to inspect the code
paths behind the reported Walsh-degree-1 sweep, verify the locked software
environment, run a minimal end-to-end calculation, and regenerate dense-sweep
outputs from source.

## Included

- `src/scalable_vqsd`: model, benchmark, trainer, and utility code.
- `experiments/ucr_method/sec5`: Walsh-degree-1 sweep and aggregation scripts.
- `uv.lock` and `pyproject.toml`: pinned environment metadata.
- `tests`: lightweight release sanity checks.
- `README.md`: reproducibility instructions and output descriptions.
- `LICENSE` and `CITATION.cff`: reuse and citation metadata.

## Excluded

- Generated result directories and figures.
- Restart checkpoint outputs.
- Local virtual environments.
- Profiler traces and machine-local logs.
- Large local archives.

The excluded artifacts are not needed to inspect or rerun the source workflow.
They are omitted to keep the repository auditable and portable.

## Boundary of This Artifact

The repository covers the compact CPU/PennyLane reproduction path for the
Weyl--Heisenberg Walsh-degree experiment. If the manuscript availability
statement refers to additional GPU-only runs, larger `d=16` checks, or archived
numerical result files, those artifacts should be listed separately alongside
this source repository.

For practical verification, reviewers can start with:

```bash
uv sync --frozen
uv run pytest
```

and then run the minimal end-to-end command documented in `README.md`.
