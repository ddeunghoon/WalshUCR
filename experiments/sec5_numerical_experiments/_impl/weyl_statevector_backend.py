from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from walsh_ucr.models.vqsd import FullUcrVQSD


jax.config.update("jax_enable_x64", True)


SUPPORTED_SIMULATION_BACKENDS = ("pennylane", "jax_statevector")


def normalize_simulation_backend(value: str | None) -> str:
    backend = "pennylane" if value is None else str(value).strip().lower()
    if backend not in SUPPORTED_SIMULATION_BACKENDS:
        supported = ", ".join(SUPPORTED_SIMULATION_BACKENDS)
        raise ValueError(f"Unsupported simulation_backend '{value}'. Supported values: {supported}.")
    return backend


@dataclass(frozen=True)
class StatevectorContext:
    n_wires: int
    n_sys: int
    n_anc: int
    indices: jax.Array
    bits: jax.Array
    bit_masks: tuple[int, ...]
    qft: jax.Array


@lru_cache(maxsize=None)
def _context(n_sys: int, n_anc: int) -> StatevectorContext:
    n_sys = int(n_sys)
    n_anc = int(n_anc)
    n_wires = n_sys + n_anc
    dim = 1 << n_wires
    indices = jnp.arange(dim, dtype=jnp.uint32)
    shifts = jnp.asarray([n_wires - 1 - wire for wire in range(n_wires)], dtype=jnp.uint32)
    bits = ((indices[:, None] >> shifts[None, :]) & jnp.asarray(1, dtype=jnp.uint32)).astype(jnp.bool_)
    bit_masks = tuple(1 << (n_wires - 1 - wire) for wire in range(n_wires))
    return StatevectorContext(
        n_wires=n_wires,
        n_sys=n_sys,
        n_anc=n_anc,
        indices=indices,
        bits=bits,
        bit_masks=bit_masks,
        qft=_qft_matrix(1 << n_sys),
    )


def _qft_matrix(dim: int) -> jax.Array:
    idx = jnp.arange(int(dim), dtype=jnp.float64)
    phase = 2j * jnp.pi * idx[:, None] * idx[None, :] / float(dim)
    return jnp.exp(phase) / math.sqrt(float(dim))


def _initial_state(ctx: StatevectorContext) -> jax.Array:
    return jnp.zeros((1 << ctx.n_wires,), dtype=jnp.complex128).at[0].set(1.0 + 0.0j)


def _control_mask(ctx: StatevectorContext, controls: Sequence[int], values: Sequence[int]) -> jax.Array:
    if not controls:
        return jnp.ones((1 << ctx.n_wires,), dtype=jnp.bool_)
    mask = jnp.ones((1 << ctx.n_wires,), dtype=jnp.bool_)
    for control, value in zip(controls, values, strict=True):
        mask = mask & (ctx.bits[:, int(control)] == bool(int(value)))
    return mask


def _apply_ry(
    state: jax.Array,
    angle: jax.Array,
    *,
    target: int,
    ctx: StatevectorContext,
    controls: Sequence[int] = (),
    control_values: Sequence[int] = (),
) -> jax.Array:
    mask = _control_mask(ctx, controls, control_values)
    target = int(target)
    bit = ctx.bits[:, target]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[target], dtype=ctx.indices.dtype)
    other = state[partner]
    angle = jnp.asarray(angle)
    c = jnp.cos(angle / 2.0)
    s = jnp.sin(angle / 2.0)
    rotated = jnp.where(bit, s * other + c * state, c * state - s * other)
    return jnp.where(mask, rotated, state)


def _apply_rz(
    state: jax.Array,
    angle: jax.Array,
    *,
    target: int,
    ctx: StatevectorContext,
    controls: Sequence[int] = (),
    control_values: Sequence[int] = (),
) -> jax.Array:
    mask = _control_mask(ctx, controls, control_values)
    bit = ctx.bits[:, int(target)]
    half = jnp.asarray(angle) / 2.0
    phase = jnp.where(bit, jnp.exp(1j * half), jnp.exp(-1j * half))
    return jnp.where(mask, state * phase, state)


def _apply_h(state: jax.Array, *, target: int, ctx: StatevectorContext) -> jax.Array:
    target = int(target)
    bit = ctx.bits[:, target]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[target], dtype=ctx.indices.dtype)
    other = state[partner]
    inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=jnp.float64)
    transformed = jnp.where(bit, (other - state) * inv_sqrt2, (state + other) * inv_sqrt2)
    return transformed


def _apply_cnot(state: jax.Array, *, control: int, target: int, ctx: StatevectorContext) -> jax.Array:
    mask = ctx.bits[:, int(control)]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[int(target)], dtype=ctx.indices.dtype)
    return jnp.where(mask, state[partner], state)


def _apply_cz(
    state: jax.Array,
    *,
    wire0: int,
    wire1: int,
    ctx: StatevectorContext,
    controls: Sequence[int] = (),
    control_values: Sequence[int] = (),
) -> jax.Array:
    mask = _control_mask(ctx, controls, control_values)
    active = mask & ctx.bits[:, int(wire0)] & ctx.bits[:, int(wire1)]
    return jnp.where(active, -state, state)


def _apply_system_unitary(state: jax.Array, matrix: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    sys_dim = 1 << ctx.n_sys
    anc_dim = 1 << ctx.n_anc
    state_2d = state.reshape((sys_dim, anc_dim))
    return (matrix @ state_2d).reshape((-1,))


def _apply_zd(state: jax.Array, power: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    d = 1 << ctx.n_sys
    power = jnp.mod(jnp.asarray(power, dtype=jnp.float64), float(d))
    for wire in range(ctx.n_sys):
        weight = 2 ** (ctx.n_sys - 1 - wire)
        angle = 2.0 * jnp.pi * power * float(weight) / float(d)
        state = _apply_rz(state, angle, target=wire, ctx=ctx)
    return state


def _apply_xd(state: jax.Array, power: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    state = _apply_system_unitary(state, ctx.qft, ctx=ctx)
    state = _apply_zd(state, power, ctx=ctx)
    state = _apply_system_unitary(state, jnp.conjugate(ctx.qft.T), ctx=ctx)
    return state


def _seed_from_angles(state: jax.Array, seed_angles: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    for wire in range(ctx.n_sys):
        state = _apply_ry(state, seed_angles[wire, 0], target=wire, ctx=ctx)
        state = _apply_rz(state, seed_angles[wire, 1], target=wire, ctx=ctx)
    for wire in range(ctx.n_sys - 1):
        state = _apply_cnot(state, control=wire, target=wire + 1, ctx=ctx)
    if ctx.n_sys > 1:
        state = _apply_cz(state, wire0=ctx.n_sys - 1, wire1=0, ctx=ctx)
    return state


def _simple_scrambler(state: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    for wire in range(ctx.n_sys):
        if wire % 2 == 0:
            state = _apply_h(state, target=wire, ctx=ctx)
        else:
            state = _apply_rz(state, jnp.pi / 2.0, target=wire, ctx=ctx)
    for wire in range(ctx.n_sys - 1):
        state = _apply_cz(state, wire0=wire, wire1=wire + 1, ctx=ctx)
    return state


def _prepare_weyl_state(
    *,
    a: jax.Array,
    b: jax.Array,
    seed_angles: jax.Array,
    use_scrambler: bool,
    ctx: StatevectorContext,
) -> jax.Array:
    state = _initial_state(ctx)
    state = _seed_from_angles(state, seed_angles, ctx=ctx)
    state = _apply_zd(state, b, ctx=ctx)
    state = _apply_xd(state, a, ctx=ctx)
    if bool(use_scrambler):
        state = _simple_scrambler(state, ctx=ctx)
    return state


def _su_block_wires(n_sys: int, num_cz: int) -> list[tuple[int, int]]:
    if int(n_sys) < 2:
        return []
    block_wires = 2 * np.arange(int(num_cz)) % (int(n_sys) - 1)
    if (int(n_sys) % 2) and int(num_cz) > 0:
        block_wires = block_wires + (np.arange(int(num_cz)) // (int(n_sys) // 2) % 2)
    return [(int(wire), int(wire) + 1) for wire in block_wires.astype(int).tolist()]


def _apply_su_ansatz(
    state: jax.Array,
    params: jax.Array,
    *,
    model: FullUcrVQSD,
    ctx: StatevectorContext,
    controls: Sequence[int] = (),
    control_values: Sequence[int] = (),
) -> jax.Array:
    n_sys = int(model.n_sys)
    su_dim, params_init, params_per_cz, num_cz, num_final_params = model.ansatz_specs(
        n_sys,
        depth=model.su_depth,
    )
    if int(params.shape[0]) != int(su_dim):
        raise ValueError(f"SU ansatz expects {su_dim} parameters, got {params.shape[0]}.")

    for offset, wire in enumerate(range(n_sys)):
        state = _apply_rz(state, params[offset], target=wire, ctx=ctx, controls=controls, control_values=control_values)
    for offset, wire in enumerate(range(n_sys), start=n_sys):
        state = _apply_ry(state, params[offset], target=wire, ctx=ctx, controls=controls, control_values=control_values)
    for offset, wire in enumerate(range(n_sys), start=2 * n_sys):
        state = _apply_rz(state, params[offset], target=wire, ctx=ctx, controls=controls, control_values=control_values)

    if int(num_cz) == 0:
        return state

    idx = int(params_init)
    for block_idx, (wire0, wire1) in enumerate(_su_block_wires(n_sys, int(num_cz))):
        use_n = int(params_per_cz if block_idx < int(num_cz) - 1 else num_final_params)
        state = _apply_cz(
            state,
            wire0=wire0,
            wire1=wire1,
            ctx=ctx,
            controls=controls,
            control_values=control_values,
        )
        state = _apply_ry(
            state,
            params[idx],
            target=wire0,
            ctx=ctx,
            controls=controls,
            control_values=control_values,
        )
        if use_n > 1:
            state = _apply_ry(
                state,
                params[idx + 1],
                target=wire1,
                ctx=ctx,
                controls=controls,
                control_values=control_values,
            )
            if use_n > 2:
                state = _apply_rz(
                    state,
                    params[idx + 2],
                    target=wire0,
                    ctx=ctx,
                    controls=controls,
                    control_values=control_values,
                )
                if use_n > 3:
                    state = _apply_rz(
                        state,
                        params[idx + 3],
                        target=wire1,
                        ctx=ctx,
                        controls=controls,
                        control_values=control_values,
                    )
        idx += use_n
    return state


def _selector_values(index: int, num_controls: int) -> tuple[int, ...]:
    return tuple((int(index) >> (int(num_controls) - 1 - bit)) & 1 for bit in range(int(num_controls)))


def _apply_full_ucry(
    state: jax.Array,
    thetas: jax.Array,
    *,
    controls: Sequence[int],
    target: int,
    ctx: StatevectorContext,
) -> jax.Array:
    expected = 1 << len(controls)
    if int(thetas.shape[0]) != expected:
        raise ValueError(f"UCRy expects {expected} parameters, got {thetas.shape[0]}.")
    for branch_idx in range(expected):
        state = _apply_ry(
            state,
            thetas[branch_idx],
            target=target,
            ctx=ctx,
            controls=controls,
            control_values=_selector_values(branch_idx, len(controls)),
        )
    return state


def _apply_model(state: jax.Array, theta: jax.Array, *, model: FullUcrVQSD, ctx: StatevectorContext) -> jax.Array:
    params = model.layout.unpack(theta)
    sys_wires = tuple(range(ctx.n_sys))
    anc_wires = tuple(range(ctx.n_sys, ctx.n_sys + ctx.n_anc))

    state = _apply_su_ansatz(state, params.SU_0, model=model, ctx=ctx)
    state = _apply_full_ucry(state, params.UCR_0, controls=sys_wires, target=anc_wires[0], ctx=ctx)
    for block_idx in range(1, ctx.n_anc):
        mt_params = getattr(params, f"MTPLX_{block_idx}")
        prev_anc = anc_wires[:block_idx]
        branches = 1 << block_idx
        dim = int(model.su_param_dim)
        if int(mt_params.shape[0]) != branches * dim:
            raise ValueError(f"MTPLX_{block_idx} expects {branches * dim} parameters, got {mt_params.shape[0]}.")
        for branch_idx in range(branches):
            branch = mt_params[branch_idx * dim : (branch_idx + 1) * dim]
            state = _apply_su_ansatz(
                state,
                branch,
                model=model,
                ctx=ctx,
                controls=prev_anc,
                control_values=_selector_values(branch_idx, block_idx),
            )
        state = _apply_full_ucry(
            state,
            getattr(params, f"UCR_{block_idx}"),
            controls=sys_wires + prev_anc,
            target=anc_wires[block_idx],
            ctx=ctx,
        )
    return state


def _ancilla_probs(state: jax.Array, *, ctx: StatevectorContext) -> jax.Array:
    probs = jnp.abs(state.reshape((1 << ctx.n_sys, 1 << ctx.n_anc))) ** 2
    return jnp.sum(probs, axis=0)


def make_weyl_statevector_batched_qnode(
    *,
    problem: dict[str, Any],
    model: FullUcrVQSD,
    n_sys: int,
) -> Any:
    if str(problem["problem_type"]) != "weyl":
        raise ValueError("jax_statevector backend only supports Section 5 Weyl problems.")
    n_anc = int(problem["n_anc"])
    ctx = _context(int(n_sys), n_anc)
    seed_angles = jnp.asarray(problem["benchmark"].seed_angles)
    use_scrambler = bool(problem["benchmark"].use_scrambler)

    def circuit_one(a: jax.Array, b: jax.Array, params: jax.Array) -> jax.Array:
        state = _prepare_weyl_state(
            a=a,
            b=b,
            seed_angles=seed_angles,
            use_scrambler=use_scrambler,
            ctx=ctx,
        )
        state = _apply_model(state, params, model=model, ctx=ctx)
        return _ancilla_probs(state, ctx=ctx)

    def batched_qnode(inputs: Any, params: jax.Array) -> jax.Array:
        a_values, b_values = inputs
        return jax.vmap(lambda a, b: circuit_one(a, b, params))(a_values, b_values)

    return batched_qnode


__all__ = [
    "SUPPORTED_SIMULATION_BACKENDS",
    "make_weyl_statevector_batched_qnode",
    "normalize_simulation_backend",
]
