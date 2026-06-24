# Code Availability

This repository is the compact WalshUCR source-code release for the
Weyl--Heisenberg Walsh-degree experiments in the associated manuscript,
"Walsh-structured uniformly controlled rotations for variational quantum state
discrimination."

It contains:

- the `scalable_vqsd` source package;
- the Section 5 Walsh-degree-1 sweep and aggregation scripts;
- pinned `uv` environment metadata;
- license and citation metadata;
- smoke tests for release sanity checks.

It intentionally excludes:

- local virtual environments;
- generated result directories;
- restart checkpoint outputs;
- generated figures and logs;
- large local archives.

Suggested manuscript wording after the repository or archival DOI is assigned:

> The WalshUCR source code used for the Weyl--Heisenberg Walsh-degree experiments
> is available at `<repository URL or DOI>`. The repository includes the pinned
> `uv` environment, release tests, and scripts for regenerating the reported
> Walsh-degree-1 sweep outputs. Generated result files and large machine-local
> artifacts are not committed.

If this repository is used as the code artifact for the whole manuscript, list
any additional repositories or archived data needed for GPU-only or larger-scale
checks in the same availability statement.
