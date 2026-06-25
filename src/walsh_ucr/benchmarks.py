from abc import ABC, abstractmethod
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml


class Benchmark(ABC):
    @abstractmethod
    def get_circuit_fn(self):
        """
        Returns a function `circuit_fn(inputs, n_qubits)` that applies the
        state preparation circuit.
        `inputs` contains whatever data is needed for the circuit (e.g., indices, angles).
        """
        pass

    @abstractmethod
    def generate_data(self, key, n_samples):
        """
        Returns `inputs` (for circuit) and `targets` (for loss calculation).
        """
        pass

    @property
    @abstractmethod
    def vmap_axes(self):
        """
        Returns the `in_axes` tuple for `jax.vmap` corresponding to the `inputs`
        structure returned by `generate_data`.
        """
        pass


def apply_Zd(power, n_qubits: int):
    d = 2**n_qubits
    power = jnp.mod(power, d)
    for i in range(n_qubits):
        weight = 2 ** (n_qubits - 1 - i)
        angle = 2 * jnp.pi * power * weight / d
        qml.RZ(angle, wires=i)

def apply_Xd(power, n_qubits: int):
    qml.QFT(wires=range(n_qubits))
    apply_Zd(power, n_qubits)
    qml.adjoint(qml.QFT)(wires=range(n_qubits))

def apply_Weyl(a, b, n_qubits: int):
    apply_Zd(b, n_qubits)
    apply_Xd(a, n_qubits)

def seed_from_angles(seed_angles):
    n_qubits = seed_angles.shape[0]
    for i in range(n_qubits):
        qml.RY(seed_angles[i, 0], wires=i)
        qml.RZ(seed_angles[i, 1], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    if n_qubits > 1:
        qml.CZ(wires=[n_qubits - 1, 0])

def simple_scrambler(n_qubits: int):
    for i in range(n_qubits):
        if i % 2 == 0:
            qml.Hadamard(wires=i)
        else:
            qml.RZ(jnp.pi / 2,wires=i)
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i + 1])

def make_seed_angles(key, n_qubits):
    return jax.random.uniform(key, (n_qubits, 2), minval=-jnp.pi, maxval=jnp.pi)

def sample_labels_ab_jax(key, m, n_qubits, unique=True):
    d = 1 << n_qubits
    total = d * d
    if unique:
        k = jax.random.choice(key, total, (m,), replace=False)
    else:
        k = jax.random.randint(key, (m,), 0, total)
    a = (k // d).astype(jnp.int32)
    b = (k %  d).astype(jnp.int32)
    return jnp.stack([a, b], axis=1)


def make_regular_simplex_d4_states():
    M = 5
    dtype = jnp.asarray(1.0 + 0.0j).dtype
    omega = jnp.exp(2j * jnp.pi / M)
    states = []
    for k in range(M):
        psi = jnp.array([omega ** (j * k) for j in range(1, 5)], dtype=dtype) / 2.0
        states.append(psi)
    return jnp.stack(states, axis=0)


def sample_exact_haar_state(
    *,
    rng: np.random.Generator,
    dimension: int = 8,
    fix_global_phase: bool = False,
    phase_atol: float = 1e-14,
) -> np.ndarray:
    if int(dimension) < 1:
        raise ValueError(f"dimension must be >= 1, got {dimension}.")

    z = rng.normal(size=int(dimension)) + 1j * rng.normal(size=int(dimension))
    norm = np.linalg.norm(z)
    if norm <= 1e-15:
        raise ValueError("Generated near-zero exact Haar state norm.")
    psi = np.asarray(z / norm, dtype=np.complex128)

    if fix_global_phase:
        nz = np.flatnonzero(np.abs(psi) > float(phase_atol))
        if len(nz) == 0:
            raise ValueError("Cannot phase-fix a zero vector.")
        psi = psi * np.exp(-1j * np.angle(psi[int(nz[0])]))
    return psi


def make_nested_exact_haar_states(
    *,
    seed: int,
    max_states: int = 12,
    dimension: int = 8,
    fix_global_phase: bool = False,
) -> np.ndarray:
    if int(max_states) < 1:
        raise ValueError(f"max_states must be >= 1, got {max_states}.")
    rng = np.random.default_rng(int(seed))
    states = [
        sample_exact_haar_state(
            rng=rng,
            dimension=int(dimension),
            fix_global_phase=bool(fix_global_phase),
        )
        for _ in range(int(max_states))
    ]
    return np.stack(states, axis=0)


def num_ancilla_for_outcomes(num_outcomes: int) -> int:
    if int(num_outcomes) < 2:
        raise ValueError(f"num_outcomes must be >= 2, got {num_outcomes}.")
    return int(math.ceil(math.log2(int(num_outcomes))))


def ensemble_seed_for_instance(*, master_seed: int, instance_id: int) -> int:
    if int(instance_id) < 0:
        raise ValueError(f"instance_id must be >= 0, got {instance_id}.")
    return int(master_seed) + int(instance_id)


def data_seed_for_instance(*, master_seed: int, instance_id: int) -> int:
    if int(instance_id) < 0:
        raise ValueError(f"instance_id must be >= 0, got {instance_id}.")
    return int(master_seed) + 100_000 + int(instance_id)


def build_nested_haar_problem(
    *,
    benchmark_type: str,
    n_sys: int,
    M: int,
    instance_id: int,
    master_seed: int,
    nested_max_m: int,
    state_dtype: str = "complex64",
    fix_global_phase: bool = False,
) -> dict[str, Any]:
    """Build the reusable exact-Haar state-preparation payload."""
    benchmark_type_normalized = str(benchmark_type).strip().lower()
    n_anc = num_ancilla_for_outcomes(int(M))
    ensemble_seed = ensemble_seed_for_instance(
        master_seed=int(master_seed),
        instance_id=int(instance_id),
    )
    data_seed = data_seed_for_instance(
        master_seed=int(master_seed),
        instance_id=int(instance_id),
    )

    if benchmark_type_normalized not in {"exact", "exact_haar", "exact_haar_d8"}:
        raise ValueError(f"Unknown Haar benchmark_type: {benchmark_type}.")
    benchmark = ExactHaarD8Benchmark(
        n_qubits=int(n_sys),
        max_states=int(nested_max_m),
        state_dtype=str(state_dtype),
        fix_global_phase=bool(fix_global_phase),
    )

    benchmark.initialize(jax.random.PRNGKey(int(ensemble_seed)))
    inputs, target_states = benchmark.generate_data(jax.random.PRNGKey(int(data_seed)), int(M))
    if benchmark.state_matrix_np is None:
        raise RuntimeError("Haar benchmark did not materialize a NumPy state matrix.")

    payload = {
        "problem_type": "exact_haar_d8",
        "n_anc": int(n_anc),
        "benchmark": benchmark,
        "inputs": jnp.asarray(inputs),
        "target_states": jnp.asarray(target_states, dtype=jnp.int32),
        "states_np": np.asarray(benchmark.state_matrix_np[: int(M)], dtype=np.complex128),
        "state_jnp_dtype": jnp.asarray(inputs).dtype,
        "ensemble_seed": int(ensemble_seed),
        "data_seed": int(data_seed),
    }
    if benchmark.seed is not None:
        payload["benchmark_internal_seed"] = int(benchmark.seed)
    return payload


class WeylBenchmark(Benchmark):
    def __init__(self, n_qubits, use_scrambler=True):
        self.n_qubits = n_qubits
        self.use_scrambler = use_scrambler
        self.seed_angles = None

    def initialize(self, key):
        self.seed_angles = make_seed_angles(key, self.n_qubits).astype(jnp.float32)

    def get_circuit_fn(self):
        def _circuit(inputs, n_qubits):
            a, b = inputs
            seed_from_angles(self.seed_angles)
            apply_Weyl(a, b, n_qubits)
            if self.use_scrambler:
                simple_scrambler(n_qubits)
        return _circuit

    def generate_data(self, key, n_samples):
        labels_ab = sample_labels_ab_jax(key, n_samples, self.n_qubits, unique=True)
        a = labels_ab[:, 0]
        b = labels_ab[:, 1]
        return (a, b), jnp.arange(n_samples, dtype=jnp.int32)

    @property
    def vmap_axes(self):
        return (0, 0)


class RegularSimplexD4Benchmark(Benchmark):
    def __init__(self, n_qubits=2, use_scrambler=False):
        if int(n_qubits) != 2:
            raise ValueError(f"RegularSimplexD4Benchmark requires n_qubits=2, got {n_qubits}.")
        self.n_qubits = int(n_qubits)
        self.use_scrambler = bool(use_scrambler)
        self.n_states = 5
        self.dimension = 4
        self.priors = jnp.ones((self.n_states,), dtype=jnp.float32) / self.n_states
        self.seed_angles = None
        self.state_matrix = None
        self.rhos = None

    def initialize(self, key):
        self.state_matrix = make_regular_simplex_d4_states()
        self.rhos = jnp.einsum("bi,bj->bij", self.state_matrix, jnp.conj(self.state_matrix))
        if self.use_scrambler:
            self.seed_angles = make_seed_angles(key, self.n_qubits)

    def get_circuit_fn(self):
        def _circuit(inputs, n_qubits):
            if int(n_qubits) != self.n_qubits:
                raise ValueError(
                    f"RegularSimplexD4Benchmark requires n_qubits={self.n_qubits}, got {n_qubits}."
                )
            qml.StatePrep(inputs, wires=range(n_qubits))
            if self.use_scrambler:
                if self.seed_angles is None:
                    raise RuntimeError("Benchmark is not initialized. Call `initialize(key)` first.")
                seed_from_angles(self.seed_angles)
        return _circuit

    def generate_data(self, key, n_samples):
        if self.state_matrix is None:
            raise RuntimeError("Benchmark is not initialized. Call `initialize(key)` first.")

        _ = key  # keep interface compatibility with other benchmarks
        labels = jnp.arange(n_samples, dtype=jnp.int32) % self.n_states
        inputs = self.state_matrix[labels]
        return inputs, labels

    @property
    def vmap_axes(self):
        return 0


class ExactHaarD8Benchmark(Benchmark):
    def __init__(
        self,
        n_qubits=3,
        max_states=12,
        state_dtype="complex64",
        fix_global_phase=False,
    ):
        if int(n_qubits) != 3:
            raise ValueError(f"ExactHaarD8Benchmark requires n_qubits=3, got {n_qubits}.")
        if int(max_states) < 1:
            raise ValueError(f"max_states must be >= 1, got {max_states}.")
        if str(state_dtype) not in {"complex64", "complex128"}:
            raise ValueError("state_dtype must be 'complex64' or 'complex128'.")

        self.n_qubits = int(n_qubits)
        self.dimension = 2 ** self.n_qubits
        self.max_states = int(max_states)
        self.state_dtype = str(state_dtype)
        self.fix_global_phase = bool(fix_global_phase)
        self.ensemble_name = "Exact-Haar-D8"
        self.state_generation = "normalized_complex_gaussian"
        self.stateprep_operation = "qml.StatePrep"
        self.stateprep_decomposition = "MottonenStatePreparation"
        self.basis_order = "PennyLane/StatePrep computational basis order"
        self.state_matrix = None
        self.state_matrix_np = None
        self.priors = None
        self.seed = None

    def initialize(self, key):
        seed_int = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        self.seed = seed_int
        states_np = make_nested_exact_haar_states(
            seed=seed_int,
            max_states=self.max_states,
            dimension=self.dimension,
            fix_global_phase=self.fix_global_phase,
        )
        self.state_matrix_np = states_np
        dtype = jnp.complex64 if self.state_dtype == "complex64" else jnp.complex128
        self.state_matrix = jnp.asarray(states_np, dtype=dtype)
        self.priors = jnp.ones((self.max_states,), dtype=jnp.float32) / float(self.max_states)

    def get_circuit_fn(self):
        def _circuit(inputs, n_qubits):
            if int(n_qubits) != self.n_qubits:
                raise ValueError(
                    f"ExactHaarD8Benchmark requires n_qubits={self.n_qubits}, got {n_qubits}."
                )
            qml.StatePrep(
                inputs,
                wires=range(n_qubits),
                normalize=False,
                validate_norm=False,
            )
        return _circuit

    def generate_data(self, key, n_samples):
        if self.state_matrix is None:
            raise RuntimeError("Benchmark is not initialized. Call `initialize(key)` first.")
        if int(n_samples) < 1 or int(n_samples) > self.max_states:
            raise ValueError(f"n_samples must be in [1, {self.max_states}], got {n_samples}.")

        _ = key
        labels = jnp.arange(int(n_samples), dtype=jnp.int32)
        return self.state_matrix[: int(n_samples)], labels

    @property
    def vmap_axes(self):
        return 0

