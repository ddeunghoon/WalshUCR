from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone VQSD experiment on CUDA JAX.")
    parser.add_argument("--n-sys", type=int, default=5)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--instance-id", type=int, default=5)
    parser.add_argument(
        "--model-type",
        choices=["walsh_degree_1", "walsh_degree_4", "walsh_degree_5", "full_ucr"],
        default="walsh_degree_1",
        help="Ansatz to train. The default preserves the existing Walsh degree-1 GPU experiment.",
    )
    parser.add_argument(
        "--state-family",
        choices=["weyl", "haar"],
        default="weyl",
        help="Input state ensemble. The default preserves the existing Weyl-Heisenberg experiment.",
    )
    parser.add_argument("--su-depth", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--num-restarts", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--scale-init", type=float, default=1.0)
    parser.add_argument("--bias-scale-init", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-sdp", action="store_true")
    parser.add_argument(
        "--schedule-checkpoint-chunk-size",
        type=int,
        default=32,
        help=(
            "Apply the long gate schedule in rematerialized chunks of this many ops. "
            "Smaller values reduce activation memory and increase runtime. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=4,
        help=(
            "Compute loss/gradient over chunks of the M-state batch and average gradients. "
            "This reduces batch activation memory. Use 0 or M to disable."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np


REAL_DTYPE = jnp.float32
COMPLEX_DTYPE = jnp.complex64

MODEL_ARTIFACT_PREFIX = {
    "walsh_degree_1": "wh_md_walsh_degree1_gpu",
    "walsh_degree_4": "wh_md_walsh_degree4_gpu",
    "walsh_degree_5": "wh_md_walsh_degree5_gpu",
    "full_ucr": "wh_md_full_ucr_gpu",
}

MODEL_NAMES = {
    "walsh_degree_1": "walsh_k_local",
    "walsh_degree_4": "walsh_k_local",
    "walsh_degree_5": "walsh_k_local",
    "full_ucr": "vqsd",
}


@dataclass(frozen=True)
class Context:
    n_sys: int
    n_anc: int
    n_wires: int
    dim: int
    indices: jax.Array
    bits: jax.Array
    bit_masks: tuple[int, ...]


@dataclass(frozen=True)
class Schedule:
    theta_dim: int
    kinds: jax.Array
    theta_indices: jax.Array
    partners: jax.Array
    bits: jax.Array
    active_masks: jax.Array

    @property
    def num_ops(self) -> int:
        return int(self.kinds.shape[0])


def seed_pair_for_instance(*, n_sys: int, M: int, instance_id: int) -> tuple[int, int]:
    return (
        100000 * int(n_sys) + 1000 * int(M) + int(instance_id),
        200000 * int(n_sys) + 1000 * int(M) + int(instance_id),
    )


def n_anc_for_M(M: int) -> int:
    return int(math.ceil(math.log2(int(M))))


def ansatz_specs(n: int, depth: int | None) -> tuple[int, int, int, int, int]:
    n = int(n)
    full_dim = 4**n - 1
    params_init = 3 * n
    if n == 1:
        return params_init, params_init, 0, 0, 0

    params_per_cz = 4
    full_num_cz = int(np.ceil((full_dim - params_init) / params_per_cz)) if full_dim > params_init else 0
    if depth is None:
        num_cz = full_num_cz
        su_dim = full_dim
        num_final_params = full_dim - params_init - (max(num_cz - 1, 0)) * params_per_cz
        return int(su_dim), params_init, params_per_cz, num_cz, int(num_final_params)

    num_cz = int(max(0, min(int(depth), full_num_cz)))
    su_dim = params_init + num_cz * params_per_cz
    return int(su_dim), params_init, params_per_cz, num_cz, params_per_cz


def make_context(n_sys: int, n_anc: int) -> Context:
    n_wires = int(n_sys) + int(n_anc)
    dim = 1 << n_wires
    indices = jnp.arange(dim, dtype=jnp.uint32)
    shifts = jnp.asarray([n_wires - 1 - wire for wire in range(n_wires)], dtype=jnp.uint32)
    bits = ((indices[:, None] >> shifts[None, :]) & jnp.asarray(1, dtype=jnp.uint32)).astype(jnp.bool_)
    bit_masks = tuple(1 << (n_wires - 1 - wire) for wire in range(n_wires))
    return Context(int(n_sys), int(n_anc), n_wires, dim, indices, bits, bit_masks)


def qft_matrix(dim: int) -> jax.Array:
    idx = jnp.arange(int(dim), dtype=REAL_DTYPE)
    phase = (2j * jnp.pi * idx[:, None] * idx[None, :] / jnp.asarray(float(dim), dtype=REAL_DTYPE)).astype(
        COMPLEX_DTYPE
    )
    return (jnp.exp(phase) / jnp.sqrt(jnp.asarray(float(dim), dtype=REAL_DTYPE))).astype(COMPLEX_DTYPE)


def apply_system_ry(state: jax.Array, angle: jax.Array, *, target: int, ctx: Context) -> jax.Array:
    bit = ctx.bits[:, int(target)]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[int(target)], dtype=ctx.indices.dtype)
    other = state[:, partner]
    c = jnp.cos(angle / jnp.asarray(2.0, dtype=REAL_DTYPE))
    s = jnp.sin(angle / jnp.asarray(2.0, dtype=REAL_DTYPE))
    return jnp.where(bit[None, :], s * other + c * state, c * state - s * other).astype(COMPLEX_DTYPE)


def apply_system_rz_batch(state: jax.Array, angles: jax.Array, *, target: int, ctx: Context) -> jax.Array:
    bit = ctx.bits[:, int(target)]
    half = jnp.asarray(angles, dtype=REAL_DTYPE) / jnp.asarray(2.0, dtype=REAL_DTYPE)
    phase = jnp.where(
        bit[None, :],
        jnp.exp((1j * half[:, None]).astype(COMPLEX_DTYPE)),
        jnp.exp((-1j * half[:, None]).astype(COMPLEX_DTYPE)),
    )
    return (state * phase).astype(COMPLEX_DTYPE)


def apply_system_h(state: jax.Array, *, target: int, ctx: Context) -> jax.Array:
    bit = ctx.bits[:, int(target)]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[int(target)], dtype=ctx.indices.dtype)
    other = state[:, partner]
    inv_sqrt2 = jnp.asarray(1.0 / math.sqrt(2.0), dtype=REAL_DTYPE)
    return jnp.where(bit[None, :], (other - state) * inv_sqrt2, (state + other) * inv_sqrt2).astype(COMPLEX_DTYPE)


def apply_system_cnot(state: jax.Array, *, control: int, target: int, ctx: Context) -> jax.Array:
    active = ctx.bits[:, int(control)]
    partner = ctx.indices ^ jnp.asarray(ctx.bit_masks[int(target)], dtype=ctx.indices.dtype)
    return jnp.where(active[None, :], state[:, partner], state).astype(COMPLEX_DTYPE)


def apply_system_cz(state: jax.Array, *, wire0: int, wire1: int, ctx: Context) -> jax.Array:
    active = ctx.bits[:, int(wire0)] & ctx.bits[:, int(wire1)]
    return jnp.where(active[None, :], -state, state).astype(COMPLEX_DTYPE)


def apply_system_zd(state: jax.Array, powers: jax.Array, *, ctx: Context) -> jax.Array:
    d = 1 << ctx.n_sys
    powers = jnp.mod(jnp.asarray(powers, dtype=REAL_DTYPE), jnp.asarray(float(d), dtype=REAL_DTYPE))
    for wire in range(ctx.n_sys):
        weight = 2 ** (ctx.n_sys - 1 - wire)
        angles = jnp.asarray(2.0 * math.pi, dtype=REAL_DTYPE) * powers * float(weight) / float(d)
        state = apply_system_rz_batch(state, angles, target=wire, ctx=ctx)
    return state


def apply_system_xd(state: jax.Array, powers: jax.Array, *, ctx: Context) -> jax.Array:
    qft = qft_matrix(1 << ctx.n_sys)
    state = (state @ qft.T).astype(COMPLEX_DTYPE)
    state = apply_system_zd(state, powers, ctx=ctx)
    return (state @ jnp.conjugate(qft)).astype(COMPLEX_DTYPE)


def make_weyl_states(
    *,
    n_sys: int,
    n_anc: int,
    M: int,
    benchmark_seed: int,
    data_seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    sys_ctx = make_context(int(n_sys), 0)
    sys_dim = 1 << int(n_sys)
    anc_dim = 1 << int(n_anc)
    full_dim = sys_dim * anc_dim
    key_seed = jax.random.PRNGKey(int(benchmark_seed))
    key_data = jax.random.PRNGKey(int(data_seed))
    seed_angles = jax.random.uniform(key_seed, (int(n_sys), 2), minval=-jnp.pi, maxval=jnp.pi, dtype=REAL_DTYPE)

    labels = jax.random.choice(key_data, sys_dim * sys_dim, (int(M),), replace=False)
    a_values = (labels // sys_dim).astype(jnp.int32)
    b_values = (labels % sys_dim).astype(jnp.int32)

    state = jnp.zeros((int(M), sys_dim), dtype=COMPLEX_DTYPE).at[:, 0].set(
        jnp.asarray(1.0 + 0.0j, dtype=COMPLEX_DTYPE)
    )
    for wire in range(int(n_sys)):
        state = apply_system_ry(state, seed_angles[wire, 0], target=wire, ctx=sys_ctx)
        state = apply_system_rz_batch(state, jnp.full((int(M),), seed_angles[wire, 1]), target=wire, ctx=sys_ctx)
    for wire in range(int(n_sys) - 1):
        state = apply_system_cnot(state, control=wire, target=wire + 1, ctx=sys_ctx)
    if int(n_sys) > 1:
        state = apply_system_cz(state, wire0=int(n_sys) - 1, wire1=0, ctx=sys_ctx)

    state = apply_system_zd(state, b_values, ctx=sys_ctx)
    state = apply_system_xd(state, a_values, ctx=sys_ctx)

    for wire in range(int(n_sys)):
        if wire % 2 == 0:
            state = apply_system_h(state, target=wire, ctx=sys_ctx)
        else:
            state = apply_system_rz_batch(
                state,
                jnp.full((int(M),), jnp.asarray(math.pi / 2.0, dtype=REAL_DTYPE), dtype=REAL_DTYPE),
                target=wire,
                ctx=sys_ctx,
            )
    for wire in range(int(n_sys) - 1):
        state = apply_system_cz(state, wire0=wire, wire1=wire + 1, ctx=sys_ctx)

    full_state = jnp.zeros((int(M), full_dim), dtype=COMPLEX_DTYPE)
    full_indices = jnp.arange(sys_dim, dtype=jnp.int32) * int(anc_dim)
    full_state = full_state.at[:, full_indices].set(state)
    return full_state, jnp.arange(int(M), dtype=jnp.int32), a_values, b_values


def make_haar_states(
    *,
    n_sys: int,
    n_anc: int,
    M: int,
    state_seed: int,
) -> tuple[jax.Array, jax.Array]:
    sys_dim = 1 << int(n_sys)
    anc_dim = 1 << int(n_anc)
    full_dim = sys_dim * anc_dim
    rng = np.random.Generator(np.random.PCG64(int(state_seed)))
    real = rng.normal(size=(int(M), sys_dim))
    imag = rng.normal(size=(int(M), sys_dim))
    state_np = real + 1j * imag
    norms = np.linalg.norm(state_np, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise ValueError("Encountered near-zero norm while generating Haar states.")
    state_np = (state_np / norms).astype(np.complex64, copy=False)
    state = jnp.asarray(state_np, dtype=COMPLEX_DTYPE)

    full_state = jnp.zeros((int(M), full_dim), dtype=COMPLEX_DTYPE)
    full_indices = jnp.arange(sys_dim, dtype=jnp.int32) * int(anc_dim)
    full_state = full_state.at[:, full_indices].set(state)
    return full_state, jnp.arange(int(M), dtype=jnp.int32)


def control_mask(ctx: Context, controls: Sequence[int], values: Sequence[int]) -> np.ndarray:
    if not controls:
        return np.ones((ctx.dim,), dtype=np.bool_)
    bits_np = np.asarray(ctx.bits)
    mask = np.ones((ctx.dim,), dtype=np.bool_)
    for control, value in zip(controls, values, strict=True):
        mask &= bits_np[:, int(control)] == bool(int(value))
    return mask


def selector_values(index: int, num_controls: int) -> tuple[int, ...]:
    return tuple((int(index) >> (int(num_controls) - 1 - bit)) & 1 for bit in range(int(num_controls)))


def su_block_wires(n_sys: int, num_cz: int) -> list[tuple[int, int]]:
    if int(n_sys) < 2:
        return []
    block_wires = 2 * np.arange(int(num_cz)) % (int(n_sys) - 1)
    if (int(n_sys) % 2) and int(num_cz) > 0:
        block_wires = block_wires + (np.arange(int(num_cz)) // (int(n_sys) // 2) % 2)
    return [(int(wire), int(wire) + 1) for wire in block_wires.astype(int).tolist()]


class ScheduleBuilder:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.kinds: list[int] = []
        self.theta_indices: list[int] = []
        self.partners: list[np.ndarray] = []
        self.bits: list[np.ndarray] = []
        self.active_masks: list[np.ndarray] = []
        self.theta_means: list[float] = []
        self.theta_scales: list[float] = []
        self._arange_partner = np.arange(ctx.dim, dtype=np.uint32)
        self._false_bits = np.zeros((ctx.dim,), dtype=np.bool_)

    def add_param(self, *, mean: float, scale: float) -> int:
        idx = len(self.theta_means)
        self.theta_means.append(float(mean))
        self.theta_scales.append(float(scale))
        return idx

    def _append_op(self, *, kind: int, theta_index: int, target: int = 0, active: np.ndarray | None = None) -> None:
        self.kinds.append(int(kind))
        self.theta_indices.append(int(theta_index))
        if kind in (0, 1, 3):
            partner = np.asarray(self.ctx.indices ^ jnp.asarray(self.ctx.bit_masks[int(target)], dtype=self.ctx.indices.dtype))
            bit = np.asarray(self.ctx.bits[:, int(target)])
        else:
            partner = self._arange_partner
            bit = self._false_bits
        if active is None:
            active = np.ones((self.ctx.dim,), dtype=np.bool_)
        self.partners.append(partner.astype(np.uint32, copy=False))
        self.bits.append(bit.astype(np.bool_, copy=False))
        self.active_masks.append(active.astype(np.bool_, copy=False))

    def add_ry_param(
        self,
        *,
        target: int,
        controls: Sequence[int] = (),
        control_values: Sequence[int] = (),
        mean: float,
        scale: float,
    ) -> None:
        idx = self.add_param(mean=mean, scale=scale)
        self._append_op(kind=0, theta_index=idx, target=target, active=control_mask(self.ctx, controls, control_values))

    def add_rz_param(
        self,
        *,
        target: int,
        controls: Sequence[int] = (),
        control_values: Sequence[int] = (),
        mean: float,
        scale: float,
    ) -> None:
        idx = self.add_param(mean=mean, scale=scale)
        self._append_op(kind=1, theta_index=idx, target=target, active=control_mask(self.ctx, controls, control_values))

    def add_cz(self, *, wire0: int, wire1: int, controls: Sequence[int] = (), control_values: Sequence[int] = ()) -> None:
        active = (
            control_mask(self.ctx, controls, control_values)
            & np.asarray(self.ctx.bits[:, int(wire0)])
            & np.asarray(self.ctx.bits[:, int(wire1)])
        )
        self._append_op(kind=2, theta_index=-1, active=active)

    def add_cnot(self, *, control: int, target: int) -> None:
        active = np.asarray(self.ctx.bits[:, int(control)])
        self._append_op(kind=3, theta_index=-1, target=target, active=active)

    def build(self) -> tuple[Schedule, jax.Array, jax.Array]:
        return (
            Schedule(
                theta_dim=len(self.theta_means),
                kinds=jnp.asarray(self.kinds, dtype=jnp.int32),
                theta_indices=jnp.asarray(self.theta_indices, dtype=jnp.int32),
                partners=jnp.asarray(np.stack(self.partners, axis=0), dtype=jnp.uint32),
                bits=jnp.asarray(np.stack(self.bits, axis=0), dtype=jnp.bool_),
                active_masks=jnp.asarray(np.stack(self.active_masks, axis=0), dtype=jnp.bool_),
            ),
            jnp.asarray(self.theta_means, dtype=REAL_DTYPE),
            jnp.asarray(self.theta_scales, dtype=REAL_DTYPE),
        )


def add_su_ansatz(builder: ScheduleBuilder, *, n_sys: int, su_depth: int, controls: Sequence[int] = (), control_values: Sequence[int] = ()) -> None:
    _, params_init, params_per_cz, num_cz, num_final_params = ansatz_specs(n_sys, depth=su_depth)
    _ = params_init
    for wire in range(int(n_sys)):
        builder.add_rz_param(target=wire, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
    for wire in range(int(n_sys)):
        builder.add_ry_param(target=wire, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
    for wire in range(int(n_sys)):
        builder.add_rz_param(target=wire, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
    for block_idx, (wire0, wire1) in enumerate(su_block_wires(n_sys, int(num_cz))):
        use_n = int(params_per_cz if block_idx < int(num_cz) - 1 else num_final_params)
        builder.add_cz(wire0=wire0, wire1=wire1, controls=controls, control_values=control_values)
        builder.add_ry_param(target=wire0, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
        if use_n > 1:
            builder.add_ry_param(target=wire1, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
            if use_n > 2:
                builder.add_rz_param(target=wire0, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)
                if use_n > 3:
                    builder.add_rz_param(target=wire1, controls=controls, control_values=control_values, mean=math.pi / 2.0, scale=0.01)


def add_walsh_k_local_ucry(
    builder: ScheduleBuilder,
    *,
    controls: Sequence[int],
    target: int,
    degree: int,
    scale_init: float,
) -> None:
    ctrls = tuple(int(control) for control in controls)
    for local_degree in range(1, min(int(degree), len(ctrls)) + 1):
        for ctrl_group in combinations(ctrls, local_degree):
            for control in ctrl_group:
                builder.add_cnot(control=int(control), target=int(target))
            builder.add_ry_param(target=int(target), mean=0.0, scale=float(scale_init))
            for control in reversed(ctrl_group):
                builder.add_cnot(control=int(control), target=int(target))


def add_walsh_degree1_ucry(builder: ScheduleBuilder, *, controls: Sequence[int], target: int, scale_init: float) -> None:
    add_walsh_k_local_ucry(
        builder,
        controls=controls,
        target=target,
        degree=1,
        scale_init=scale_init,
    )


def build_walsh_k_local_schedule(
    ctx: Context,
    *,
    su_depth: int,
    degree: int,
    scale_init: float,
    bias_scale_init: float,
) -> tuple[Schedule, jax.Array, jax.Array]:
    builder = ScheduleBuilder(ctx)
    sys_wires = tuple(range(ctx.n_sys))
    anc_wires = tuple(range(ctx.n_sys, ctx.n_sys + ctx.n_anc))
    add_su_ansatz(builder, n_sys=ctx.n_sys, su_depth=su_depth)
    builder.add_ry_param(target=anc_wires[0], mean=math.pi / 2.0, scale=float(bias_scale_init))
    add_walsh_k_local_ucry(
        builder,
        controls=sys_wires,
        target=anc_wires[0],
        degree=degree,
        scale_init=scale_init,
    )
    for block_idx in range(1, ctx.n_anc):
        prev_anc = anc_wires[:block_idx]
        for branch_idx in range(1 << block_idx):
            add_su_ansatz(
                builder,
                n_sys=ctx.n_sys,
                su_depth=su_depth,
                controls=prev_anc,
                control_values=selector_values(branch_idx, block_idx),
            )
        target = anc_wires[block_idx]
        builder.add_ry_param(target=target, mean=math.pi / 2.0, scale=float(bias_scale_init))
        add_walsh_k_local_ucry(
            builder,
            controls=sys_wires + prev_anc,
            target=target,
            degree=degree,
            scale_init=scale_init,
        )
    return builder.build()


def build_walsh_degree1_schedule(
    ctx: Context,
    *,
    su_depth: int,
    scale_init: float,
    bias_scale_init: float,
) -> tuple[Schedule, jax.Array, jax.Array]:
    return build_walsh_k_local_schedule(
        ctx,
        su_depth=su_depth,
        degree=1,
        scale_init=scale_init,
        bias_scale_init=bias_scale_init,
    )


def build_full_ucr_schedule(
    ctx: Context,
    *,
    su_depth: int,
    scale_init: float,
) -> tuple[Schedule, jax.Array, jax.Array]:
    builder = ScheduleBuilder(ctx)
    sys_wires = tuple(range(ctx.n_sys))
    anc_wires = tuple(range(ctx.n_sys, ctx.n_sys + ctx.n_anc))
    add_su_ansatz(builder, n_sys=ctx.n_sys, su_depth=su_depth)
    for branch_idx in range(1 << ctx.n_sys):
        builder.add_ry_param(
            target=anc_wires[0],
            controls=sys_wires,
            control_values=selector_values(branch_idx, ctx.n_sys),
            mean=math.pi / 2.0,
            scale=float(scale_init),
        )
    for block_idx in range(1, ctx.n_anc):
        prev_anc = anc_wires[:block_idx]
        for branch_idx in range(1 << block_idx):
            add_su_ansatz(
                builder,
                n_sys=ctx.n_sys,
                su_depth=su_depth,
                controls=prev_anc,
                control_values=selector_values(branch_idx, block_idx),
            )
        ctrls = sys_wires + prev_anc
        for branch_idx in range(1 << len(ctrls)):
            builder.add_ry_param(
                target=anc_wires[block_idx],
                controls=ctrls,
                control_values=selector_values(branch_idx, len(ctrls)),
                mean=math.pi / 2.0,
                scale=float(scale_init),
            )
    return builder.build()


def build_model_schedule(
    model_type: str,
    ctx: Context,
    *,
    su_depth: int,
    scale_init: float,
    bias_scale_init: float,
) -> tuple[Schedule, jax.Array, jax.Array]:
    if model_type == "walsh_degree_1":
        return build_walsh_k_local_schedule(
            ctx,
            su_depth=su_depth,
            degree=1,
            scale_init=scale_init,
            bias_scale_init=bias_scale_init,
        )
    if model_type == "walsh_degree_4":
        return build_walsh_k_local_schedule(
            ctx,
            su_depth=su_depth,
            degree=4,
            scale_init=scale_init,
            bias_scale_init=bias_scale_init,
        )
    if model_type == "walsh_degree_5":
        return build_walsh_k_local_schedule(
            ctx,
            su_depth=su_depth,
            degree=5,
            scale_init=scale_init,
            bias_scale_init=bias_scale_init,
        )
    if model_type == "full_ucr":
        return build_full_ucr_schedule(
            ctx,
            su_depth=su_depth,
            scale_init=scale_init,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def model_artifact_prefix(model_type: str, *, state_family: str = "weyl") -> str:
    try:
        prefix = MODEL_ARTIFACT_PREFIX[str(model_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type: {model_type}") from exc
    if str(state_family) == "weyl":
        return prefix
    if str(state_family) == "haar":
        return prefix.replace("wh_md_", "haar_", 1)
    raise ValueError(f"Unsupported state_family: {state_family}")


def model_name_for_type(model_type: str) -> str:
    try:
        return MODEL_NAMES[str(model_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type: {model_type}") from exc


def model_walsh_degree(model_type: str) -> int | None:
    if str(model_type) == "walsh_degree_1":
        return 1
    if str(model_type) == "walsh_degree_4":
        return 4
    if str(model_type) == "walsh_degree_5":
        return 5
    return None


def init_theta(key: jax.Array, means: jax.Array, scales: jax.Array) -> jax.Array:
    return (means + scales * jax.random.normal(key, means.shape, dtype=REAL_DTYPE)).astype(REAL_DTYPE)


def apply_schedule(
    state: jax.Array,
    theta: jax.Array,
    schedule: Schedule,
    *,
    checkpoint_chunk_size: int,
) -> jax.Array:
    def step(current: jax.Array, op: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, None]:
        kind, theta_idx, partner, bit, active = op
        safe_theta_idx = jnp.maximum(theta_idx, jnp.asarray(0, dtype=jnp.int32))
        angle = theta[safe_theta_idx]
        partner_i = partner.astype(jnp.int32)
        other = current[:, partner_i]
        c = jnp.cos(angle / jnp.asarray(2.0, dtype=REAL_DTYPE))
        s = jnp.sin(angle / jnp.asarray(2.0, dtype=REAL_DTYPE))
        ry_next = jnp.where(bit[None, :], s * other + c * current, c * current - s * other)
        half = angle / jnp.asarray(2.0, dtype=REAL_DTYPE)
        phase = jnp.where(bit, jnp.exp((1j * half).astype(COMPLEX_DTYPE)), jnp.exp((-1j * half).astype(COMPLEX_DTYPE)))
        rz_next = current * phase[None, :]
        cz_next = jnp.where(active[None, :], -current, current)
        cnot_next = jnp.where(active[None, :], other, current)

        def do_ry(_: None) -> jax.Array:
            return jnp.where(active[None, :], ry_next, current)

        def do_rz(_: None) -> jax.Array:
            return jnp.where(active[None, :], rz_next, current)

        def do_cz(_: None) -> jax.Array:
            return cz_next

        def do_cnot(_: None) -> jax.Array:
            return cnot_next

        next_state = jax.lax.switch(kind, (do_ry, do_rz, do_cz, do_cnot), None).astype(COMPLEX_DTYPE)
        return next_state, None

    ops = (schedule.kinds, schedule.theta_indices, schedule.partners, schedule.bits, schedule.active_masks)
    chunk_size = int(checkpoint_chunk_size)
    if chunk_size <= 0:
        final, _ = jax.lax.scan(step, state, ops)
        return final

    num_ops = int(schedule.kinds.shape[0])
    pad_len = (-num_ops) % chunk_size
    if pad_len:
        dim = int(schedule.partners.shape[1])
        pad_kinds = jnp.full((pad_len,), 2, dtype=schedule.kinds.dtype)
        pad_theta_indices = jnp.zeros((pad_len,), dtype=schedule.theta_indices.dtype)
        pad_partners = jnp.broadcast_to(jnp.arange(dim, dtype=schedule.partners.dtype), (pad_len, dim))
        pad_bits = jnp.zeros((pad_len, dim), dtype=schedule.bits.dtype)
        pad_active_masks = jnp.zeros((pad_len, dim), dtype=schedule.active_masks.dtype)
        kinds = jnp.concatenate([schedule.kinds, pad_kinds], axis=0)
        theta_indices = jnp.concatenate([schedule.theta_indices, pad_theta_indices], axis=0)
        partners = jnp.concatenate([schedule.partners, pad_partners], axis=0)
        bits = jnp.concatenate([schedule.bits, pad_bits], axis=0)
        active_masks = jnp.concatenate([schedule.active_masks, pad_active_masks], axis=0)
    else:
        kinds = schedule.kinds
        theta_indices = schedule.theta_indices
        partners = schedule.partners
        bits = schedule.bits
        active_masks = schedule.active_masks

    num_chunks = int(kinds.shape[0]) // chunk_size
    chunked_ops = (
        kinds.reshape((num_chunks, chunk_size)),
        theta_indices.reshape((num_chunks, chunk_size)),
        partners.reshape((num_chunks, chunk_size, partners.shape[1])),
        bits.reshape((num_chunks, chunk_size, bits.shape[1])),
        active_masks.reshape((num_chunks, chunk_size, active_masks.shape[1])),
    )

    def chunk_step(
        current: jax.Array,
        chunk_ops: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        next_state, _ = jax.lax.scan(step, current, chunk_ops)
        return next_state, None

    final, _ = jax.lax.scan(jax.checkpoint(chunk_step), state, chunked_ops, _split_transpose=True)
    return final


def ancilla_probs(state: jax.Array, *, ctx: Context) -> jax.Array:
    probs = jnp.abs(state.reshape((state.shape[0], 1 << ctx.n_sys, 1 << ctx.n_anc))) ** 2
    return jnp.sum(probs, axis=1)


def make_train_step(
    *,
    states: jax.Array,
    targets: jax.Array,
    ctx: Context,
    schedule: Schedule,
    learning_rate: float,
    M: int,
    schedule_checkpoint_chunk_size: int,
    microbatch_size: int,
) -> Any:
    lr = jnp.asarray(float(learning_rate), dtype=REAL_DTYPE)
    beta1 = jnp.asarray(0.9, dtype=REAL_DTYPE)
    beta2 = jnp.asarray(0.999, dtype=REAL_DTYPE)
    eps = jnp.asarray(1e-8, dtype=REAL_DTYPE)
    batch_size = int(states.shape[0])
    requested_microbatch_size = int(microbatch_size)
    use_microbatch = requested_microbatch_size > 0 and requested_microbatch_size < batch_size
    if use_microbatch:
        if batch_size % requested_microbatch_size != 0:
            raise ValueError(
                f"microbatch_size={requested_microbatch_size} must divide M={batch_size}. "
                "Use 0, M, or a divisor of M."
            )
        effective_microbatch_size = requested_microbatch_size
    else:
        effective_microbatch_size = batch_size
    num_microbatches = batch_size // effective_microbatch_size
    batch_indices = jnp.arange(effective_microbatch_size, dtype=jnp.int32)
    micro_states = states.reshape((num_microbatches, effective_microbatch_size) + tuple(states.shape[1:]))
    micro_targets = targets.reshape((num_microbatches, effective_microbatch_size))

    def micro_loss_fn(params: jax.Array, mb_states: jax.Array, mb_targets: jax.Array) -> jax.Array:
        final_state = apply_schedule(
            mb_states,
            params,
            schedule,
            checkpoint_chunk_size=int(schedule_checkpoint_chunk_size),
        )
        probs = ancilla_probs(final_state, ctx=ctx)[:, : int(M)]
        p_correct = probs[batch_indices, mb_targets]
        return (jnp.asarray(1.0, dtype=REAL_DTYPE) - jnp.mean(p_correct)).astype(REAL_DTYPE)

    def loss_and_grad(params: jax.Array) -> tuple[jax.Array, jax.Array]:
        if num_microbatches == 1:
            return jax.value_and_grad(micro_loss_fn)(params, micro_states[0], micro_targets[0])

        def scan_body(
            carry: tuple[jax.Array, jax.Array],
            mb: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            loss_acc, grad_acc = carry
            mb_states, mb_targets = mb
            mb_loss, mb_grad = jax.value_and_grad(micro_loss_fn)(params, mb_states, mb_targets)
            return (loss_acc + mb_loss, grad_acc + mb_grad), None

        init = (jnp.asarray(0.0, dtype=REAL_DTYPE), jnp.zeros_like(params))
        (loss_sum, grad_sum), _ = jax.lax.scan(scan_body, init, (micro_states, micro_targets))
        denom = jnp.asarray(float(num_microbatches), dtype=REAL_DTYPE)
        return loss_sum / denom, grad_sum / denom

    @jax.jit
    def train_step(
        params: jax.Array,
        m: jax.Array,
        v: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        loss, grad = loss_and_grad(params)
        t_next = t + jnp.asarray(1, dtype=jnp.int32)
        m_next = beta1 * m + (jnp.asarray(1.0, dtype=REAL_DTYPE) - beta1) * grad
        v_next = beta2 * v + (jnp.asarray(1.0, dtype=REAL_DTYPE) - beta2) * (grad * grad)
        t_float = t_next.astype(REAL_DTYPE)
        m_hat = m_next / (jnp.asarray(1.0, dtype=REAL_DTYPE) - beta1**t_float)
        v_hat = v_next / (jnp.asarray(1.0, dtype=REAL_DTYPE) - beta2**t_float)
        params_next = (params - lr * m_hat / (jnp.sqrt(v_hat) + eps)).astype(REAL_DTYPE)
        return params_next, m_next, v_next, t_next, loss, jnp.linalg.norm(grad)

    return train_step


def make_output_dir(
    cli_output_dir: str | None,
    *,
    n_sys: int,
    M: int,
    instance_id: int,
    num_restarts: int,
    su_depth: int,
    model_type: str,
    state_family: str,
) -> Path:
    if cli_output_dir:
        output_dir = Path(cli_output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = model_artifact_prefix(model_type, state_family=state_family)
        output_dir = (
            Path(__file__).resolve().parent
            / "results"
            / f"{prefix}_nsys{int(n_sys)}_M{int(M)}_instance{int(instance_id):02d}_r{int(num_restarts)}_su{int(su_depth)}"
            / stamp
        )
    (output_dir / "raw" / "restart_checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    return output_dir


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Encountered non-finite value: {result}")
    return result


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(contiguous.shape)).encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def resolve_project_root(required_dir: str) -> Path:
    env_root = os.environ.get("QSD_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / required_dir).exists():
            return root
        raise FileNotFoundError(f"QSD_PROJECT_ROOT={root} does not contain {required_dir}")
    return next(
        (parent for parent in Path(__file__).resolve().parents if (parent / required_dir).exists()),
        Path(__file__).resolve().parents[2],
    )


def compute_sdp_optimum(*, n_sys: int, M: int, instance_id: int) -> float:
    project_root = resolve_project_root("WalshUCR")
    walshucr_root = project_root / "WalshUCR"
    if not walshucr_root.exists():
        raise FileNotFoundError(f"Expected WalshUCR project at {walshucr_root}")

    code = """
import json
import sys
sys.path.append('WalshUCR/experiments/sec5_numerical_experiments/_impl')
from wh_d8_sweep import _problem_namespace, _seed_pair_for_instance, compute_optimum_success_probability
from weyl_problem import _build_problem_instance
import jax
n_sys = int(__NSYS__)
M = int(__M__)
instance_id = int(__INSTANCE__)
benchmark_seed, data_seed = _seed_pair_for_instance(n_sys=n_sys, M=M, instance_id=instance_id)
args = _problem_namespace(
    n_sys=n_sys,
    m_outcome=M,
    benchmark_seed=benchmark_seed,
    data_seed=data_seed,
    optimizer='adam',
    learning_rate=1e-2,
    steps=1,
    eval_interval=1,
    threshold=1e-6,
    tol=5e-4,
    su_depth=1,
    scale_init=1.0,
    bias_scale_init=1.0,
    weight_decay=0.0,
    state_dtype='complex128',
)
# Match the GPU experiment's Weyl seed-state generation.  wh_md_sweep enables
# x64 globally at import time, but JAX random.uniform produces different
# float32 values under x64=True vs x64=False for the same PRNGKey.
jax.config.update("jax_enable_x64", False)
problem = _build_problem_instance(args)
jax.config.update("jax_enable_x64", True)
p_opt = compute_optimum_success_probability(problem=problem, n_sys=n_sys, m_outcome=M)
print(json.dumps({'p_opt_sdp': float(p_opt)}))
""".replace("__NSYS__", str(int(n_sys))).replace("__M__", str(int(M))).replace("__INSTANCE__", str(int(instance_id)))

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["JAX_PLATFORM_NAME"] = "cpu"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    completed = subprocess.run(
        ["uv", "run", "--project", str(walshucr_root), "python", "-c", code],
        cwd=str(project_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return float(json.loads(stripped)["p_opt_sdp"])
        except (json.JSONDecodeError, KeyError):
            continue
    raise RuntimeError(f"Could not parse p_opt_sdp from SDP output:\n{completed.stdout}\n{completed.stderr}")


def compute_haar_sdp_optimum(*, n_sys: int, M: int, state_seed: int) -> float:
    project_root = resolve_project_root("WalshUCR")
    walshucr_root = project_root / "WalshUCR"
    if not walshucr_root.exists():
        raise FileNotFoundError(f"Expected WalshUCR project at {walshucr_root}")

    code = """
import json
import sys
sys.path.append('WalshUCR/src')
from walsh_ucr.utils.haar import generate_haar_states, make_weighted_pure_state_rhos
from walsh_ucr.utils.sdp import sdp_med

n_sys = int(__NSYS__)
M = int(__M__)
state_seed = int(__STATE_SEED__)
dim = 1 << n_sys
states = generate_haar_states(seed=state_seed, num_states=M, dim=dim)
q_rhos = make_weighted_pure_state_rhos(states)
sdp_error, _ = sdp_med(q_rhos, M, num_povm=M)
print(json.dumps({'p_opt_sdp': float(1.0 - float(sdp_error))}))
""".replace("__NSYS__", str(int(n_sys))).replace("__M__", str(int(M))).replace(
        "__STATE_SEED__", str(int(state_seed))
    )

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["JAX_PLATFORM_NAME"] = "cpu"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    completed = subprocess.run(
        ["uv", "run", "--project", str(walshucr_root), "python", "-c", code],
        cwd=str(project_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return float(json.loads(stripped)["p_opt_sdp"])
        except (json.JSONDecodeError, KeyError):
            continue
    raise RuntimeError(f"Could not parse p_opt_sdp from Haar SDP output:\n{completed.stdout}\n{completed.stderr}")


def format_restart_line(row: dict[str, Any], *, p_opt_sdp: float | None) -> str:
    p_succ = float(row["p_succ"])
    gap = float("nan") if p_opt_sdp is None else float(p_opt_sdp - p_succ)
    return (
        f"{int(row['restart_id'])},"
        f"{gap:.6g},"
        f"{float(row['compile_plus_first_step_sec']):.1f},"
        f"{float(row['wall_clock_sec']):.1f},"
        f"{int(row['num_steps'])}"
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_completed_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["restart_id"])] = row
    return rows


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_restart(
    *,
    restart_id: int,
    model_type: str,
    model_name: str,
    schedule: Schedule,
    means: jax.Array,
    scales: jax.Array,
    train_step: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key = jax.random.PRNGKey(int(args.seed_start) + int(restart_id))
    theta = init_theta(key, means, scales)
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    t = jnp.asarray(0, dtype=jnp.int32)

    start = time.perf_counter()
    theta, m, v, t, loss, grad_norm = train_step(theta, m, v, t)
    jax.block_until_ready(theta)
    compile_plus_first_step_sec = time.perf_counter() - start

    losses = [finite_float(loss)]
    grad_norms = [finite_float(grad_norm)]
    eval_records: list[dict[str, Any]] = [
        {
            "restart_id": int(restart_id),
            "seed_opt": int(args.seed_start) + int(restart_id),
            "step": 1,
            "objective_value": float(losses[0]),
            "best_objective_value": float(losses[0]),
            "p_succ": float(1.0 - losses[0]),
            "best_p_succ": float(1.0 - losses[0]),
            "grad_norm": float(grad_norms[0]),
            "elapsed_sec": float(compile_plus_first_step_sec),
        }
    ]
    termination_reason = "max_steps"
    remaining_steps = max(0, int(args.steps) - 1)
    post_start = time.perf_counter()
    last_eval_loss = losses[0]
    for step_idx in range(2, int(args.steps) + 1):
        theta, m, v, t, loss, grad_norm = train_step(theta, m, v, t)
        jax.block_until_ready(theta)
        current_loss = finite_float(loss)
        losses.append(current_loss)
        current_grad_norm = finite_float(grad_norm)
        grad_norms.append(current_grad_norm)
        if int(args.eval_interval) > 0 and step_idx % int(args.eval_interval) == 0:
            best_loss_so_far = min(losses)
            eval_records.append(
                {
                    "restart_id": int(restart_id),
                    "seed_opt": int(args.seed_start) + int(restart_id),
                    "step": int(step_idx),
                    "objective_value": float(current_loss),
                    "best_objective_value": float(best_loss_so_far),
                    "p_succ": float(1.0 - current_loss),
                    "best_p_succ": float(1.0 - best_loss_so_far),
                    "grad_norm": float(current_grad_norm),
                    "elapsed_sec": float(compile_plus_first_step_sec + time.perf_counter() - post_start),
                }
            )
            if abs(last_eval_loss - current_loss) <= float(args.threshold):
                termination_reason = "threshold"
                break
            last_eval_loss = current_loss
    post_compile_sec = time.perf_counter() - post_start

    final_loss = losses[-1]
    best_loss = min(losses)
    if eval_records[-1]["step"] != len(losses):
        eval_records.append(
            {
                "restart_id": int(restart_id),
                "seed_opt": int(args.seed_start) + int(restart_id),
                "step": int(len(losses)),
                "objective_value": float(final_loss),
                "best_objective_value": float(best_loss),
                "p_succ": float(1.0 - final_loss),
                "best_p_succ": float(1.0 - best_loss),
                "grad_norm": float(grad_norms[-1]),
                "elapsed_sec": float(compile_plus_first_step_sec + post_compile_sec),
            }
        )
    row = {
        "model_type": str(model_type),
        "model_name": str(model_name),
        "restart_id": int(restart_id),
        "seed_opt": int(args.seed_start) + int(restart_id),
        "num_steps": int(len(losses)),
        "termination_reason": str(termination_reason),
        "theta_dim": int(schedule.theta_dim),
        "num_ops": int(schedule.num_ops),
        "initial_loss": float(losses[0]),
        "final_objective_value": float(final_loss),
        "best_objective_value": float(best_loss),
        "p_succ": float(1.0 - final_loss),
        "grad_norm_last": float(grad_norms[-1]),
        "compile_plus_first_step_sec": float(compile_plus_first_step_sec),
        "post_compile_total_sec": float(post_compile_sec),
        "post_compile_sec_per_step": float(post_compile_sec / max(1, len(losses) - 1)) if remaining_steps else 0.0,
        "wall_clock_sec": float(compile_plus_first_step_sec + post_compile_sec),
        "theta": json.dumps(np.asarray(theta, dtype=np.float32).reshape(-1).tolist()),
    }
    return row, eval_records


def main() -> None:
    args = ARGS
    if int(args.n_sys) < 1:
        raise ValueError("--n-sys must be >= 1.")
    if int(args.M) < 2:
        raise ValueError("--M must be >= 2.")
    state_family = str(args.state_family)
    if state_family == "weyl" and int(args.M) > 4 ** int(args.n_sys):
        raise ValueError("--M must be <= 4**n_sys for unique Weyl labels.")
    if int(args.steps) < 1:
        raise ValueError("--steps must be >= 1.")
    if int(args.num_restarts) < 1:
        raise ValueError("--num-restarts must be >= 1.")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if int(args.schedule_checkpoint_chunk_size) < 0:
        raise ValueError("--schedule-checkpoint-chunk-size must be >= 0.")
    if int(args.microbatch_size) < 0:
        raise ValueError("--microbatch-size must be >= 0.")
    if int(args.microbatch_size) not in (0, int(args.M)) and int(args.microbatch_size) > 0:
        if int(args.M) % int(args.microbatch_size) != 0:
            raise ValueError("--microbatch-size must divide --M, or use 0/M to disable microbatching.")
    model_type = str(args.model_type)
    model_name = model_name_for_type(model_type)
    artifact_prefix = model_artifact_prefix(model_type, state_family=state_family)

    devices = jax.devices()
    default_backend = jax.default_backend()
    if bool(args.require_gpu) and default_backend != "gpu":
        raise RuntimeError(f"Expected JAX default backend 'gpu', got {default_backend}. devices={devices}")

    n_anc = n_anc_for_M(int(args.M))
    raw_outcomes = 1 << n_anc
    if raw_outcomes < int(args.M):
        raise ValueError("Internal error: raw_outcomes must cover M.")
    ctx = make_context(int(args.n_sys), n_anc)
    benchmark_seed, data_seed = seed_pair_for_instance(
        n_sys=int(args.n_sys),
        M=int(args.M),
        instance_id=int(args.instance_id),
    )
    state_seed = int(data_seed)
    if state_family == "weyl":
        states, targets, a_values, b_values = make_weyl_states(
            n_sys=int(args.n_sys),
            n_anc=n_anc,
            M=int(args.M),
            benchmark_seed=benchmark_seed,
            data_seed=data_seed,
        )
    elif state_family == "haar":
        states, targets = make_haar_states(
            n_sys=int(args.n_sys),
            n_anc=n_anc,
            M=int(args.M),
            state_seed=state_seed,
        )
        a_values = None
        b_values = None
    else:
        raise ValueError(f"Unsupported state_family: {state_family}")
    states.block_until_ready()
    state_norms = jnp.linalg.norm(states, axis=1)
    state_norms.block_until_ready()
    max_norm_error = float(jnp.max(jnp.abs(state_norms - jnp.asarray(1.0, dtype=REAL_DTYPE))))
    state_array_sha256 = array_sha256(np.asarray(jax.device_get(states)))

    schedule, means, scales = build_model_schedule(
        model_type,
        ctx,
        su_depth=int(args.su_depth),
        scale_init=float(args.scale_init),
        bias_scale_init=float(args.bias_scale_init),
    )
    output_dir = make_output_dir(
        args.output_dir,
        n_sys=int(args.n_sys),
        M=int(args.M),
        instance_id=int(args.instance_id),
        num_restarts=int(args.num_restarts),
        su_depth=int(args.su_depth),
        model_type=model_type,
        state_family=state_family,
    )
    restart_jsonl_path = output_dir / "raw" / f"{artifact_prefix}_restart_records.jsonl"
    eval_jsonl_path = output_dir / "raw" / f"{artifact_prefix}_eval_records.jsonl"
    eval_csv_path = output_dir / "raw" / f"{artifact_prefix}_eval_records.csv"
    checkpoint_model_tag = f"{model_type}_gpu" if state_family == "weyl" else f"{model_type}_{state_family}_gpu"
    checkpoint_path = output_dir / "raw" / "restart_checkpoints" / (
        f"nsys{int(args.n_sys)}_M{int(args.M)}_instance{int(args.instance_id):02d}_{checkpoint_model_tag}.jsonl"
    )
    results_csv_path = output_dir / "raw" / f"{artifact_prefix}_results.csv"
    summary_json_path = output_dir / "summaries" / f"{artifact_prefix}_summary.json"
    completed_rows = load_completed_rows(checkpoint_path)

    if bool(args.skip_sdp):
        p_opt_sdp = None
    elif state_family == "weyl":
        p_opt_sdp = compute_sdp_optimum(
            n_sys=int(args.n_sys),
            M=int(args.M),
            instance_id=int(args.instance_id),
        )
    elif state_family == "haar":
        p_opt_sdp = compute_haar_sdp_optimum(
            n_sys=int(args.n_sys),
            M=int(args.M),
            state_seed=state_seed,
        )
    else:
        raise ValueError(f"Unsupported state_family: {state_family}")
    if p_opt_sdp is None:
        print("p_opt_sdp=nan", flush=True)
    else:
        print(f"p_opt_sdp={p_opt_sdp:.8f}", flush=True)
    print(f"state_family={state_family}", flush=True)
    print(f"model_type={model_type} model_name={model_name} theta_dim={schedule.theta_dim} num_ops={schedule.num_ops}", flush=True)
    print(
        f"schedule_checkpoint_chunk_size={int(args.schedule_checkpoint_chunk_size)} "
        f"microbatch_size={int(args.microbatch_size)}",
        flush=True,
    )
    print("restart_id,gap,compile_s,finish_s,step", flush=True)

    train_step = make_train_step(
        states=states,
        targets=targets,
        ctx=ctx,
        schedule=schedule,
        learning_rate=float(args.learning_rate),
        M=int(args.M),
        schedule_checkpoint_chunk_size=int(args.schedule_checkpoint_chunk_size),
        microbatch_size=int(args.microbatch_size),
    )

    rows: list[dict[str, Any]] = [completed_rows[key] for key in sorted(completed_rows)]
    for restart_id in range(int(args.num_restarts)):
        if restart_id in completed_rows:
            print(format_restart_line(completed_rows[restart_id], p_opt_sdp=p_opt_sdp), flush=True)
            continue
        row, eval_records = run_restart(
            restart_id=restart_id,
            model_type=model_type,
            model_name=model_name,
            schedule=schedule,
            means=means,
            scales=scales,
            train_step=train_step,
            args=args,
        )
        metadata = {
            "instance_id": int(args.instance_id),
            "n_sys": int(args.n_sys),
            "d": int(1 << int(args.n_sys)),
            "M": int(args.M),
            "M_over_d": float(int(args.M) / float(1 << int(args.n_sys))),
            "n_anc": int(n_anc),
            "raw_outcomes": int(raw_outcomes),
            "effective_m_outcomes": int(args.M),
            "projection_strategy": "drop_extra",
            "coverage_ratio": float(int(args.M) / float(raw_outcomes)),
            "state_family": str(state_family),
            "model_type": str(model_type),
            "model_name": str(model_name),
            "walsh_degree": model_walsh_degree(model_type),
            "optimizer_name": "adam",
            "learning_rate": float(args.learning_rate),
            "learning_rate_schedule": "constant",
            "max_steps": int(args.steps),
            "eval_interval": int(args.eval_interval),
            "threshold": float(args.threshold),
            "num_restarts": int(args.num_restarts),
            "su_depth": int(args.su_depth),
            "scale_init": float(args.scale_init),
            "bias_scale_init": float(args.bias_scale_init),
            "memory_optimization": "schedule_checkpoint_plus_microbatch",
            "schedule_checkpoint_chunk_size": int(args.schedule_checkpoint_chunk_size),
            "microbatch_size": int(args.microbatch_size),
            "num_microbatches": int(int(args.M) // int(args.microbatch_size))
            if int(args.microbatch_size) not in (0, int(args.M))
            else 1,
            "p_opt_sdp": float(p_opt_sdp) if p_opt_sdp is not None else None,
            "dtype": "complex64",
            "theta_dtype": "float32",
            "jax_backend": str(default_backend),
        }
        state_metadata = {
            "jax_devices": ";".join(str(device) for device in devices),
            "benchmark_seed": int(benchmark_seed),
            "data_seed": int(data_seed),
            "state_seed": int(state_seed),
            "state_array_sha256": str(state_array_sha256),
        }
        if state_family == "weyl":
            state_metadata.update(
                {
                    "a_values": json.dumps([int(value) for value in np.asarray(a_values)]),
                    "b_values": json.dumps([int(value) for value in np.asarray(b_values)]),
                }
            )
        row.update(
            {
                **metadata,
                **state_metadata,
                "gap_abs_sdp": float(p_opt_sdp - float(row["p_succ"])) if p_opt_sdp is not None else None,
            }
        )
        rows.append(row)
        append_jsonl(checkpoint_path, row)
        for eval_record in eval_records:
            eval_record.update(metadata)
            eval_record.update(state_metadata)
            eval_record["gap_abs_sdp"] = (
                float(p_opt_sdp - float(eval_record["p_succ"])) if p_opt_sdp is not None else None
            )
            append_jsonl(eval_jsonl_path, eval_record)
        print(format_restart_line(row, p_opt_sdp=p_opt_sdp), flush=True)

    rows = sorted(rows, key=lambda row: int(row["restart_id"]))
    write_csv(results_csv_path, rows)
    restart_jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    eval_rows = sorted(
        load_jsonl_rows(eval_jsonl_path),
        key=lambda row: (int(row["restart_id"]), int(row["step"])),
    )
    write_csv(eval_csv_path, eval_rows)
    best = min(rows, key=lambda row: float(row["final_objective_value"])) if rows else None
    summary = {
        "config": {
            "n_sys": int(args.n_sys),
            "d": int(1 << int(args.n_sys)),
            "M": int(args.M),
            "M_over_d": float(int(args.M) / float(1 << int(args.n_sys))),
            "n_anc": int(n_anc),
            "instance_id": int(args.instance_id),
            "state_family": str(state_family),
            "su_depth": int(args.su_depth),
            "steps": int(args.steps),
            "eval_interval": int(args.eval_interval),
            "num_restarts": int(args.num_restarts),
            "seed_start": int(args.seed_start),
            "model_type": str(model_type),
            "model_name": str(model_name),
            "walsh_degree": model_walsh_degree(model_type),
            "optimizer_name": "adam",
            "learning_rate": float(args.learning_rate),
            "threshold": float(args.threshold),
            "projection_strategy": "drop_extra",
            "memory_optimization": "schedule_checkpoint_plus_microbatch",
            "schedule_checkpoint_chunk_size": int(args.schedule_checkpoint_chunk_size),
            "microbatch_size": int(args.microbatch_size),
            "num_microbatches": int(int(args.M) // int(args.microbatch_size))
            if int(args.microbatch_size) not in (0, int(args.M))
            else 1,
            "p_opt_sdp": float(p_opt_sdp) if p_opt_sdp is not None else None,
            "dtype": "complex64",
            "theta_dtype": "float32",
            "jax_backend": str(default_backend),
            "jax_devices": [str(device) for device in devices],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "state_precompute": {
            "states_shape": list(states.shape),
            "states_dtype": str(states.dtype),
            "max_norm_error": float(max_norm_error),
            "benchmark_seed": int(benchmark_seed),
            "data_seed": int(data_seed),
            "state_seed": int(state_seed),
            "state_array_sha256": str(state_array_sha256),
        },
        "model": {
            "model_type": str(model_type),
            "model_name": str(model_name),
            "theta_dim": int(schedule.theta_dim),
            "num_ops": int(schedule.num_ops),
            "best_restart": int(best["restart_id"]) if best else None,
            "seed_opt": int(best["seed_opt"]) if best else None,
            "p_succ": float(best["p_succ"]) if best else None,
            "gap_abs_sdp": float(p_opt_sdp - float(best["p_succ"])) if best and p_opt_sdp is not None else None,
            "final_objective_value": float(best["final_objective_value"]) if best else None,
            "best_objective_value": float(best["best_objective_value"]) if best else None,
            "wall_clock_sec_best_restart": float(best["wall_clock_sec"]) if best else None,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "checkpoint_jsonl": str(checkpoint_path),
            "restart_records_jsonl": str(restart_jsonl_path),
            "eval_records_jsonl": str(eval_jsonl_path),
            "eval_records_csv": str(eval_csv_path),
            "results_csv": str(results_csv_path),
            "summary_json": str(summary_json_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
