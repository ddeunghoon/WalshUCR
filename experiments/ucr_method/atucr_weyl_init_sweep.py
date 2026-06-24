from __future__ import annotations

import argparse
import math
from typing import Any

import jax
import jax.numpy as jnp

from scalable_vqsd.benchmarks import WeylBenchmark
from scalable_vqsd.utils.density_matrix import make_rhos
from scalable_vqsd.utils.sdp import sdp_med


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _validate_inputs(args: argparse.Namespace) -> int:
    if str(args.problem_type) != "weyl":
        raise ValueError("WalshUCR Section 5 release supports problem_type='weyl' only.")
    if int(args.n_sys) < 1:
        raise ValueError(f"n_sys must be >= 1, got {args.n_sys}.")
    if int(args.m_outcome) < 2:
        raise ValueError(f"m_outcome must be >= 2, got {args.m_outcome}.")
    max_unique = 4 ** int(args.n_sys)
    if int(args.m_outcome) > max_unique:
        raise ValueError(f"m_outcome must be <= 4**n_sys for unique Weyl labels: {args.m_outcome} > {max_unique}.")
    if int(args.steps) < 1:
        raise ValueError(f"steps must be >= 1, got {args.steps}.")
    if int(args.eval_interval) < 1:
        raise ValueError(f"eval_interval must be >= 1, got {args.eval_interval}.")
    if float(args.scale_init) < 0:
        raise ValueError(f"scale_init must be >= 0, got {args.scale_init}.")
    if float(args.bias_scale_init) < 0:
        raise ValueError(f"bias_scale_init must be >= 0, got {args.bias_scale_init}.")
    if float(args.weight_decay) < 0:
        raise ValueError(f"weight_decay must be >= 0, got {args.weight_decay}.")
    if str(args.optimizer) == "adam" and float(args.learning_rate) <= 0:
        raise ValueError(f"learning_rate must be > 0 for adam, got {args.learning_rate}.")
    return int(math.ceil(math.log2(int(args.m_outcome))))


def _build_problem_instance(args: argparse.Namespace) -> dict[str, Any]:
    n_anc = _validate_inputs(args)
    benchmark = WeylBenchmark(n_qubits=int(args.n_sys), use_scrambler=True)
    benchmark.initialize(jax.random.PRNGKey(int(args.benchmark_seed)))
    inputs, target_states = benchmark.generate_data(
        jax.random.PRNGKey(int(args.data_seed)),
        int(args.m_outcome),
    )
    return {
        "problem_type": "weyl",
        "n_anc": n_anc,
        "benchmark": benchmark,
        "inputs": inputs,
        "target_states": jnp.asarray(target_states, dtype=jnp.int32),
        "states_np": None,
        "state_jnp_dtype": None,
    }


def _compute_sdp_value(*, problem: dict[str, Any], n_sys: int, m_outcome: int) -> float:
    a_priori_probs = jnp.ones((int(m_outcome),), dtype=jnp.float64) / float(m_outcome)
    sys_wires = list(range(int(n_sys)))
    benchmark = problem["benchmark"]
    rhos = make_rhos(
        benchmark_type="weyl",
        inputs=problem["inputs"],
        circuit_fn=benchmark.get_circuit_fn(),
        n_sys=int(n_sys),
        sys_wires=sys_wires,
        interface="jax",
    )
    q_rhos = [a_priori_probs[idx] * rho for idx, rho in enumerate(rhos)]
    val, _ = sdp_med(q_rhos, int(m_outcome), num_povm=int(m_outcome))
    return float(val)
