from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from itertools import combinations
import math
from typing import Callable, Dict, Sequence, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: Tuple[int, ...]
    init: str = "angle"
    mean: float = 0.0
    scale: float = 1.0


class MetaParamLayout:
    """Flat theta vector to structured parameter dataclass converter."""

    def __init__(self, specs: Sequence[ParamSpec], param_cls: Type):
        self.specs = tuple(specs)
        self.names = tuple(spec.name for spec in self.specs)
        self.param_cls = param_cls

        offset = 0
        slices = {}
        for spec in self.specs:
            size = int(np.prod(spec.shape))
            slices[spec.name] = (slice(offset, offset + size), spec.shape)
            offset += size
        self.slices = slices
        self.theta_dim = int(offset)

    def unpack(self, theta_1d: jnp.ndarray):
        kwargs = {}
        for name in self.names:
            param_slice, shape = self.slices[name]
            kwargs[name] = theta_1d[param_slice].reshape(shape)
        return self.param_cls(**kwargs)

    def init_params(self, key):
        keys = jax.random.split(key, len(self.specs))
        chunks = []
        for param_key, spec in zip(keys, self.specs, strict=True):
            if spec.init == "angle":
                values = jax.random.uniform(
                    param_key,
                    shape=spec.shape,
                    minval=0.0,
                    maxval=2.0 * jnp.pi,
                )
            elif spec.init == "normal":
                values = spec.mean + spec.scale * jax.random.normal(param_key, shape=spec.shape)
            elif spec.init == "fixed":
                values = jnp.full(spec.shape, spec.mean)
            else:
                values = jnp.zeros(spec.shape)
            chunks.append(values.reshape(-1))
        return jnp.concatenate(chunks, axis=0)


@jax.tree_util.register_pytree_node_class
class VQSDLayer:
    """Recursive full-UCR VQSD layer used as the common SU/UCR base class."""

    def __init__(
        self,
        n_anc: int,
        n_sys: int,
        su_depth: int | None = 2,
        mean_init: float | str = "pi/2",
        scale_init: float = 0.01,
    ):
        self.n_anc = int(n_anc)
        self.n_sys = int(n_sys)
        self.su_depth = su_depth if su_depth is None else int(su_depth)
        mean_val = self._parse_angle(mean_init)

        su_dim, *_ = self.ansatz_specs(self.n_sys, depth=self.su_depth)
        self.su_param_dim = int(su_dim)

        specs: list[ParamSpec] = []
        fields = []

        specs.append(ParamSpec("SU_0", (self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
        fields.append(("SU_0", jnp.ndarray))
        specs.append(ParamSpec("UCR_0", (2**self.n_sys,), init="normal", mean=mean_val, scale=scale_init))
        fields.append(("UCR_0", jnp.ndarray))

        for block_idx in range(1, self.n_anc):
            mt_name = f"MTPLX_{block_idx}"
            specs.append(
                ParamSpec(mt_name, (2**block_idx * self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01)
            )
            fields.append((mt_name, jnp.ndarray))

            ucr_name = f"UCR_{block_idx}"
            specs.append(
                ParamSpec(
                    ucr_name,
                    (2 ** (self.n_sys + block_idx),),
                    init="normal",
                    mean=mean_val,
                    scale=scale_init,
                )
            )
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(f"ParamsAncilla{self.n_anc}", fields, frozen=True)
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[field_name for field_name, _ in fields],
            meta_fields=[],
        )
        self.layout = MetaParamLayout(specs, self.ParamsType)
        self._ansatz_cache: Dict[Tuple[int, ...], Callable[[jnp.ndarray], None]] = {}

    @staticmethod
    def _parse_angle(value: float | str) -> float:
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "pi/2":
                return float(jnp.pi / 2)
            if lowered == "pi":
                return float(jnp.pi)
            return float(value)
        return float(value)

    def __call__(self, theta_flat, sys_wires, anc_wires):
        params = self.layout.unpack(theta_flat)
        self.block_layer(params, list(sys_wires), list(anc_wires))

    def block_layer(self, params, sys, anc):
        if len(anc) != self.n_anc:
            raise ValueError(f"Expected {self.n_anc} ancilla wires, got {len(anc)}.")

        self.apply_su_ansatz(params.SU_0, sys)
        self.UCRy(params.UCR_0, ctrls=sys, target=anc[0])

        for block_idx in range(1, self.n_anc):
            prev_anc = anc[:block_idx]
            self.multiplexed_ansatz(getattr(params, f"MTPLX_{block_idx}"), ctrls=prev_anc, targets=sys)
            self.UCRy(getattr(params, f"UCR_{block_idx}"), ctrls=list(sys) + list(prev_anc), target=anc[block_idx])

    def _mcry_on_target(self, angle, selector_vals, ctrls, target):
        if len(selector_vals) != len(ctrls):
            raise ValueError("selector_vals length must match ctrls.")
        qml.ctrl(qml.RY, control=ctrls, control_values=tuple(selector_vals))(angle, wires=target)

    def _apply_ucry(self, thetas, ctrls, target):
        num_controls = len(ctrls)
        num_branches = 2**num_controls
        if int(thetas.shape[0]) != num_branches:
            raise ValueError(f"UCRy expects {num_branches} parameters, got {thetas.shape[0]}.")

        for branch_idx in range(num_branches):
            selector_vals = tuple((branch_idx >> (num_controls - 1 - bit)) & 1 for bit in range(num_controls))
            self._mcry_on_target(thetas[branch_idx], selector_vals, ctrls, target)

    def UCRy(self, thetas, ctrls, target):
        self._apply_ucry(thetas, ctrls, target)

    def _mcansatz_on_sys(self, params, selector_vals, ctrls, targets):
        if len(selector_vals) != len(ctrls):
            raise ValueError("selector_vals length must match ctrls.")
        qml.ctrl(
            self.apply_su_ansatz,
            control=ctrls,
            control_values=tuple(selector_vals),
        )(params, targets)

    def multiplexed_ansatz(self, thetas, ctrls, targets):
        num_controls = len(ctrls)
        num_branches = 2**num_controls
        dim = self.su_param_dim
        if int(thetas.shape[0]) != num_branches * dim:
            raise ValueError(
                f"MTPLX expects {num_branches * dim} parameters "
                f"({num_branches} branches * su_dim {dim}), got {thetas.shape[0]}."
            )

        for branch_idx in range(num_branches):
            branch_params = thetas[branch_idx * dim : (branch_idx + 1) * dim]
            selector_vals = tuple((branch_idx >> (num_controls - 1 - bit)) & 1 for bit in range(num_controls))
            self._mcansatz_on_sys(branch_params, selector_vals, ctrls, targets)

    @staticmethod
    def ansatz_specs(n: int, depth: int | None = None):
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
            return su_dim, params_init, params_per_cz, num_cz, int(num_final_params)

        num_cz = int(max(0, min(int(depth), full_num_cz)))
        su_dim = params_init + num_cz * params_per_cz
        return int(su_dim), params_init, params_per_cz, num_cz, params_per_cz

    @staticmethod
    def _init_layer_su(params, wires):
        n_wires = len(wires)
        for param, wire in zip(params[:n_wires], wires, strict=True):
            qml.RZ(param, wires=wire)
        for param, wire in zip(params[n_wires : 2 * n_wires], wires, strict=True):
            qml.RY(param, wires=wire)
        for param, wire in zip(params[2 * n_wires : 3 * n_wires], wires, strict=True):
            qml.RZ(param, wires=wire)

    @staticmethod
    def _circuit_block_su(params, wires_2):
        qml.CZ(wires_2)
        qml.RY(params[0], wires=wires_2[0])
        qml.RY(params[1], wires=wires_2[1])
        qml.RZ(params[2], wires=wires_2[0])
        qml.RZ(params[3], wires=wires_2[1])

    @staticmethod
    def _final_circuit_block_su(params, wires_2, num_params: int):
        qml.CZ(wires_2)
        qml.RY(params[0], wires=wires_2[0])
        if num_params > 1:
            qml.RY(params[1], wires=wires_2[1])
        if num_params > 2:
            qml.RZ(params[2], wires=wires_2[0])
        if num_params > 3:
            qml.RZ(params[3], wires=wires_2[1])

    def make_ansatz(self, wires):
        wires = tuple(wires)
        n_wires = len(wires)
        if n_wires < 1:
            raise ValueError("The number of wires must be positive.")

        su_dim, params_init, params_per_cz, num_cz, num_final_params = self.ansatz_specs(
            n_wires,
            depth=self.su_depth,
        )
        if n_wires == 1:
            return lambda params: self._init_layer_su(params, wires)
        if num_cz == 0:
            return lambda params: self._init_layer_su(params[:params_init], wires)

        block_wire_indices = 2 * np.arange(num_cz) % (n_wires - 1)
        if (n_wires % 2) and num_cz > 0:
            block_wire_indices = block_wire_indices + (np.arange(num_cz) // (n_wires // 2) % 2)
        block_wires = [(wires[idx], wires[idx + 1]) for idx in block_wire_indices.astype(int).tolist()]

        entries = []
        offset = params_init
        for block_idx in range(num_cz):
            use_n = params_per_cz if block_idx < num_cz - 1 else int(num_final_params)
            param_slice = slice(offset, offset + use_n)
            entries.append((param_slice, block_wires[block_idx], use_n))
            offset += use_n
        if int(su_dim) != offset:
            raise ValueError(f"Internal ansatz spec mismatch: su_dim={su_dim}, consumed={offset}.")

        def ansatz(params):
            self._init_layer_su(params[:params_init], wires)
            for param_slice, wire_pair, use_n in entries:
                if use_n == 4:
                    self._circuit_block_su(params[param_slice], wire_pair)
                else:
                    self._final_circuit_block_su(params[param_slice], wire_pair, use_n)

        return ansatz

    def _get_ansatz(self, wires):
        wires = tuple(wires)
        if wires not in self._ansatz_cache:
            self._ansatz_cache[wires] = self.make_ansatz(wires)
        return self._ansatz_cache[wires]

    def apply_su_ansatz(self, params, wires):
        self._get_ansatz(wires)(params)

    def tree_flatten(self):
        return (), (self.n_anc, self.n_sys, self.su_depth, 0.0)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_anc, n_sys, su_depth, mean_init = aux_data
        return cls(n_anc=n_anc, n_sys=n_sys, su_depth=su_depth, mean_init=mean_init)


@jax.tree_util.register_pytree_node_class
class ATUcrVQSDBias(VQSDLayer):
    """Degree-1 controlled-RY model with one bias RY per UCR block."""

    def __init__(
        self,
        n_anc: int,
        n_sys: int,
        su_depth: int | None = 2,
        mean_init: float | str = 0.0,
        scale_init: float = 0.01,
        bias_mean_init: float | str = "pi/2",
        bias_scale_init: float = 0.01,
    ):
        self.n_anc = int(n_anc)
        self.n_sys = int(n_sys)
        self.su_depth = su_depth if su_depth is None else int(su_depth)
        self.mean_init = self._parse_angle(mean_init)
        self.bias_mean_init = self._parse_angle(bias_mean_init)
        self.scale_init = float(scale_init)
        self.bias_scale_init = float(bias_scale_init)

        su_dim, *_ = self.ansatz_specs(self.n_sys, depth=self.su_depth)
        self.su_param_dim = int(su_dim)

        specs: list[ParamSpec] = []
        fields = []
        specs.append(ParamSpec("SU_0", (self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
        fields.append(("SU_0", jnp.ndarray))
        specs.append(ParamSpec("UCR_BIAS_0", (1,), init="normal", mean=self.bias_mean_init, scale=self.bias_scale_init))
        fields.append(("UCR_BIAS_0", jnp.ndarray))
        specs.append(ParamSpec("UCR_0", (self.n_sys,), init="normal", mean=self.mean_init, scale=self.scale_init))
        fields.append(("UCR_0", jnp.ndarray))

        for block_idx in range(1, self.n_anc):
            mt_name = f"MTPLX_{block_idx}"
            specs.append(
                ParamSpec(mt_name, (2**block_idx * self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01)
            )
            fields.append((mt_name, jnp.ndarray))

            bias_name = f"UCR_BIAS_{block_idx}"
            specs.append(ParamSpec(bias_name, (1,), init="normal", mean=self.bias_mean_init, scale=self.bias_scale_init))
            fields.append((bias_name, jnp.ndarray))

            ucr_name = f"UCR_{block_idx}"
            specs.append(
                ParamSpec(
                    ucr_name,
                    (self.n_sys + block_idx,),
                    init="normal",
                    mean=self.mean_init,
                    scale=self.scale_init,
                )
            )
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(f"ParamsAncilla{self.n_anc}Bias", fields, frozen=True)
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[field_name for field_name, _ in fields],
            meta_fields=[],
        )
        self.layout = MetaParamLayout(specs, self.ParamsType)
        self._ansatz_cache: Dict[Tuple[int, ...], Callable[[jnp.ndarray], None]] = {}

    def block_layer(self, params, sys, anc):
        if len(anc) != self.n_anc:
            raise ValueError(f"Expected {self.n_anc} ancilla wires, got {len(anc)}.")

        self.apply_su_ansatz(params.SU_0, sys)
        qml.RY(params.UCR_BIAS_0[0], wires=anc[0])
        self.UCRy(params.UCR_0, ctrls=sys, target=anc[0])

        for block_idx in range(1, self.n_anc):
            prev_anc = anc[:block_idx]
            self.multiplexed_ansatz(getattr(params, f"MTPLX_{block_idx}"), ctrls=prev_anc, targets=sys)
            qml.RY(getattr(params, f"UCR_BIAS_{block_idx}")[0], wires=anc[block_idx])
            self.UCRy(getattr(params, f"UCR_{block_idx}"), ctrls=list(sys) + list(prev_anc), target=anc[block_idx])

    def UCRy(self, thetas, ctrls, target):
        if len(thetas) != len(ctrls):
            raise ValueError(f"ATUcrVQSDBias.UCRy expects {len(ctrls)} parameters, got {len(thetas)}.")
        for theta, ctrl in zip(thetas, ctrls, strict=True):
            qml.ctrl(qml.RY, control=[ctrl], control_values=(1,))(theta, wires=target)

    def tree_flatten(self):
        return (), (
            self.n_anc,
            self.n_sys,
            self.su_depth,
            self.mean_init,
            self.scale_init,
            self.bias_mean_init,
            self.bias_scale_init,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_anc, n_sys, su_depth, mean_init, scale_init, bias_mean_init, bias_scale_init = aux_data
        return cls(
            n_anc=n_anc,
            n_sys=n_sys,
            su_depth=su_depth,
            mean_init=mean_init,
            scale_init=scale_init,
            bias_mean_init=bias_mean_init,
            bias_scale_init=bias_scale_init,
        )


@jax.tree_util.register_pytree_node_class
class UcrKLocalVQSD(ATUcrVQSDBias):
    """Cumulative k-local controlled-RY bias model."""

    def __init__(
        self,
        n_anc: int,
        n_sys: int,
        su_depth: int | None = 2,
        ucr_degree: int = 2,
        mean_init: float | str = 0.0,
        scale_init: float = 0.01,
        bias_mean_init: float | str = "pi/2",
        bias_scale_init: float = 0.01,
    ):
        self.ucr_degree = int(ucr_degree)
        if self.ucr_degree < 1:
            raise ValueError(f"ucr_degree must be >= 1, got {ucr_degree}.")
        super().__init__(
            n_anc=n_anc,
            n_sys=n_sys,
            su_depth=su_depth,
            mean_init=mean_init,
            scale_init=scale_init,
            bias_mean_init=bias_mean_init,
            bias_scale_init=bias_scale_init,
        )
        self._rebuild_k_local_layout()

    def _rebuild_k_local_layout(self) -> None:
        specs: list[ParamSpec] = []
        fields = []
        specs.append(ParamSpec("SU_0", (self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
        fields.append(("SU_0", jnp.ndarray))
        specs.append(ParamSpec("UCR_BIAS_0", (1,), init="normal", mean=self.bias_mean_init, scale=self.bias_scale_init))
        fields.append(("UCR_BIAS_0", jnp.ndarray))
        specs.append(
            ParamSpec(
                "UCR_0",
                (self.num_k_local_terms(self.n_sys, self.ucr_degree),),
                init="normal",
                mean=self.mean_init,
                scale=self.scale_init,
            )
        )
        fields.append(("UCR_0", jnp.ndarray))

        for block_idx in range(1, self.n_anc):
            mt_name = f"MTPLX_{block_idx}"
            specs.append(
                ParamSpec(mt_name, (2**block_idx * self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01)
            )
            fields.append((mt_name, jnp.ndarray))

            bias_name = f"UCR_BIAS_{block_idx}"
            specs.append(ParamSpec(bias_name, (1,), init="normal", mean=self.bias_mean_init, scale=self.bias_scale_init))
            fields.append((bias_name, jnp.ndarray))

            ucr_name = f"UCR_{block_idx}"
            specs.append(
                ParamSpec(
                    ucr_name,
                    (self.num_k_local_terms(self.n_sys + block_idx, self.ucr_degree),),
                    init="normal",
                    mean=self.mean_init,
                    scale=self.scale_init,
                )
            )
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(
            f"ParamsAncilla{self.n_anc}KLocalDegree{self.ucr_degree}Bias",
            fields,
            frozen=True,
        )
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[field_name for field_name, _ in fields],
            meta_fields=[],
        )
        self.layout = MetaParamLayout(specs, self.ParamsType)

    @staticmethod
    def num_k_local_terms(num_controls: int, degree: int) -> int:
        num_controls = int(num_controls)
        degree = int(degree)
        if num_controls < 0:
            raise ValueError(f"num_controls must be >= 0, got {num_controls}.")
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}.")
        return int(sum(math.comb(num_controls, local_degree) for local_degree in range(1, min(degree, num_controls) + 1)))

    def UCRy(self, thetas, ctrls, target):
        ctrls = list(ctrls)
        expected = self.num_k_local_terms(len(ctrls), self.ucr_degree)
        if len(thetas) != expected:
            raise ValueError(f"UcrKLocalVQSD.UCRy expects {expected} parameters, got {len(thetas)}.")

        theta_idx = 0
        for local_degree in range(1, min(self.ucr_degree, len(ctrls)) + 1):
            for ctrl_group in combinations(ctrls, local_degree):
                self._mcry_on_target(
                    thetas[theta_idx],
                    selector_vals=(1,) * local_degree,
                    ctrls=list(ctrl_group),
                    target=target,
                )
                theta_idx += 1

    def tree_flatten(self):
        return (), (
            self.n_anc,
            self.n_sys,
            self.su_depth,
            self.ucr_degree,
            self.mean_init,
            self.scale_init,
            self.bias_mean_init,
            self.bias_scale_init,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_anc, n_sys, su_depth, ucr_degree, mean_init, scale_init, bias_mean_init, bias_scale_init = aux_data
        return cls(
            n_anc=n_anc,
            n_sys=n_sys,
            su_depth=su_depth,
            ucr_degree=ucr_degree,
            mean_init=mean_init,
            scale_init=scale_init,
            bias_mean_init=bias_mean_init,
            bias_scale_init=bias_scale_init,
        )


@jax.tree_util.register_pytree_node_class
class WalshKLocalVQSD(UcrKLocalVQSD):
    """Walsh k-local parity-rotation UCR model."""

    @staticmethod
    def _walsh_parity_ry(theta, ctrls, target):
        ctrls = list(ctrls)
        if not ctrls:
            qml.RY(theta, wires=target)
            return

        for ctrl in ctrls:
            qml.CNOT(wires=[ctrl, target])
        qml.RY(theta, wires=target)
        for ctrl in reversed(ctrls):
            qml.CNOT(wires=[ctrl, target])

    def UCRy(self, thetas, ctrls, target):
        ctrls = list(ctrls)
        expected = self.num_k_local_terms(len(ctrls), self.ucr_degree)
        if len(thetas) != expected:
            raise ValueError(f"WalshKLocalVQSD.UCRy expects {expected} parameters, got {len(thetas)}.")

        theta_idx = 0
        for local_degree in range(1, min(self.ucr_degree, len(ctrls)) + 1):
            for ctrl_group in combinations(ctrls, local_degree):
                self._walsh_parity_ry(thetas[theta_idx], ctrls=list(ctrl_group), target=target)
                theta_idx += 1
