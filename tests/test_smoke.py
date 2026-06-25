from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from walsh_ucr.benchmarks import ExactHaarD8Benchmark, WeylBenchmark, build_nested_haar_problem
from walsh_ucr.models.vqsd import RandomSparseFullUcrVQSD, WalshKLocalVQSD


ROOT = Path(__file__).resolve().parents[1]


def test_walsh_k_local_parameter_count() -> None:
    assert WalshKLocalVQSD.num_k_local_terms(3, 1) == 3
    assert WalshKLocalVQSD.num_k_local_terms(4, 2) == 10


def test_walsh_model_initialization_shape() -> None:
    model = WalshKLocalVQSD(n_anc=1, n_sys=2, su_depth=1, ucr_degree=1)
    theta = model.layout.init_params(jax.random.PRNGKey(0))

    assert theta.shape == (model.layout.theta_dim,)
    assert model.layout.theta_dim == 13
    assert jnp.all(jnp.isfinite(theta))


def test_random_sparse_and_exact_haar_import_paths() -> None:
    model = RandomSparseFullUcrVQSD(
        n_anc=1,
        n_sys=2,
        selected_ucr_indices=((0, 3),),
        su_depth=1,
    )
    theta = model.layout.init_params(jax.random.PRNGKey(1))
    assert theta.shape == (model.layout.theta_dim,)

    benchmark = ExactHaarD8Benchmark(max_states=2)
    benchmark.initialize(jax.random.PRNGKey(2))
    inputs, targets = benchmark.generate_data(jax.random.PRNGKey(3), 2)
    assert inputs.shape == (2, 8)
    assert jnp.array_equal(targets, jnp.arange(2, dtype=jnp.int32))

    problem = build_nested_haar_problem(
        benchmark_type="exact_haar_d8",
        n_sys=3,
        M=2,
        instance_id=0,
        master_seed=20260504,
        nested_max_m=2,
    )
    assert problem["n_anc"] == 1
    assert problem["inputs"].shape == (2, 8)
    assert problem["states_np"].shape == (2, 8)
    assert problem["target_states"].shape == (2,)


def test_weyl_benchmark_data_are_deterministic() -> None:
    benchmark = WeylBenchmark(n_qubits=2, use_scrambler=True)
    benchmark.initialize(jax.random.PRNGKey(10))

    inputs_a, targets_a = benchmark.generate_data(jax.random.PRNGKey(20), 3)
    inputs_b, targets_b = benchmark.generate_data(jax.random.PRNGKey(20), 3)

    assert jnp.array_equal(inputs_a[0], inputs_b[0])
    assert jnp.array_equal(inputs_a[1], inputs_b[1])
    assert jnp.array_equal(targets_a, targets_b)
    assert targets_a.shape == (3,)


def test_release_cli_help() -> None:
    scripts = [
        ROOT / "experiments" / "sec5_numerical_experiments" / "fig_wh_d8_sweep" / "run.py",
        ROOT / "experiments" / "sec5_numerical_experiments" / "fig_haar_d8_sweep" / "run.py",
        ROOT / "experiments" / "sec5_numerical_experiments" / "fig_wh_degree_sweep" / "run.py",
        ROOT / "experiments" / "sec5_numerical_experiments" / "table_d16_checks" / "run_gpu.py",
        ROOT / "experiments" / "appendix" / "rank_diagnostics" / "run.py",
        ROOT / "data" / "validate_paper_data.py",
        ROOT / "figures" / "build_paper_figures.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert "usage:" in result.stdout


def test_readme_and_data_manifests_exist() -> None:
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "data" / "manifests" / "paper_data_manifest.json").is_file()
    assert (ROOT / "data" / "manifests" / "paper_results_manifest.toml").is_file()
