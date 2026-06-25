from __future__ import annotations

import numpy as np


def generate_haar_states(*, seed: int, num_states: int, dim: int, dtype: np.dtype = np.complex128) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    real = rng.normal(size=(int(num_states), int(dim)))
    imag = rng.normal(size=(int(num_states), int(dim)))
    states = real + 1j * imag
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError("Encountered near-zero norm while generating Haar states.")
    states = states / norms
    return states.astype(dtype, copy=False)


def make_weighted_pure_state_rhos(states: np.ndarray) -> list[np.ndarray]:
    states = np.asarray(states, dtype=np.complex128)
    num_states = int(states.shape[0])
    prior = 1.0 / float(num_states)
    return [prior * np.outer(psi, np.conj(psi)) for psi in states]
