from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from scalable_vqsd.benchmarks import WeylBenchmark
from scalable_vqsd.models.vqsd import WalshKLocalVQSD


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
    script = ROOT / "experiments" / "ucr_method" / "sec5" / "wh_md_walsh_degree1_sweep.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "Walsh degree-1" in result.stdout
    assert "--num-restarts" in result.stdout
