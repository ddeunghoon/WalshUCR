from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import pennylane as qml


class Benchmark(ABC):
    @abstractmethod
    def get_circuit_fn(self):
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, key, n_samples):
        raise NotImplementedError

    @property
    @abstractmethod
    def vmap_axes(self):
        raise NotImplementedError


def apply_Zd(power, n_qubits: int):
    d = 2**n_qubits
    power = jnp.mod(power, d)
    for wire in range(n_qubits):
        weight = 2 ** (n_qubits - 1 - wire)
        angle = 2 * jnp.pi * power * weight / d
        qml.RZ(angle, wires=wire)


def apply_Xd(power, n_qubits: int):
    qml.QFT(wires=range(n_qubits))
    apply_Zd(power, n_qubits)
    qml.adjoint(qml.QFT)(wires=range(n_qubits))


def apply_Weyl(a, b, n_qubits: int):
    apply_Zd(b, n_qubits)
    apply_Xd(a, n_qubits)


def seed_from_angles(seed_angles):
    n_qubits = seed_angles.shape[0]
    for wire in range(n_qubits):
        qml.RY(seed_angles[wire, 0], wires=wire)
        qml.RZ(seed_angles[wire, 1], wires=wire)
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    if n_qubits > 1:
        qml.CZ(wires=[n_qubits - 1, 0])


def simple_scrambler(n_qubits: int):
    for wire in range(n_qubits):
        if wire % 2 == 0:
            qml.Hadamard(wires=wire)
        else:
            qml.RZ(jnp.pi / 2, wires=wire)
    for wire in range(n_qubits - 1):
        qml.CZ(wires=[wire, wire + 1])


def make_seed_angles(key, n_qubits):
    return jax.random.uniform(key, (n_qubits, 2), minval=-jnp.pi, maxval=jnp.pi)


def sample_labels_ab_jax(key, m, n_qubits, unique=True):
    d = 1 << n_qubits
    total = d * d
    if unique:
        labels = jax.random.choice(key, total, (m,), replace=False)
    else:
        labels = jax.random.randint(key, (m,), 0, total)
    a = (labels // d).astype(jnp.int32)
    b = (labels % d).astype(jnp.int32)
    return jnp.stack([a, b], axis=1)


class WeylBenchmark(Benchmark):
    def __init__(self, n_qubits, use_scrambler=True):
        self.n_qubits = int(n_qubits)
        self.use_scrambler = bool(use_scrambler)
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
        return (labels_ab[:, 0], labels_ab[:, 1]), jnp.arange(n_samples, dtype=jnp.int32)

    @property
    def vmap_axes(self):
        return (0, 0)
