import jax
import jax.numpy as jnp
from dataclasses import dataclass, make_dataclass
from itertools import combinations
import math
from typing import Tuple, Sequence, Type, Callable, Dict
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
    """Flat theta <-> structured params 변환. (JAX array slicing/reshape 기반)"""
    def __init__(self, specs: Sequence[ParamSpec], param_cls: Type):
        self.specs = tuple(specs)
        self.names = tuple(s.name for s in self.specs)
        self.param_cls = param_cls

        offset = 0
        slices = {}
        for s in self.specs:
            size = int(np.prod(s.shape))
            sl = slice(offset, offset + size)
            slices[s.name] = (sl, s.shape)
            offset += size
        self.slices = slices
        self.theta_dim = int(offset)

    def unpack(self, theta_1d: jnp.ndarray):
        kwargs = {}
        for name in self.names:
            sl, shape = self.slices[name]
            kwargs[name] = theta_1d[sl].reshape(shape)
        return self.param_cls(**kwargs)

    def init_params(self, key):
        keys = jax.random.split(key, len(self.specs))
        chunks = []
        for k, s in zip(keys, self.specs):
            if s.init == "angle":
                x = jax.random.uniform(k, shape=s.shape, minval=0.0, maxval=2.0 * jnp.pi)
            elif s.init == "normal":
                x = s.mean + s.scale * jax.random.normal(k, shape=s.shape)
            else:
                x = jnp.zeros(s.shape)
            chunks.append(x.reshape(-1))
        return jnp.concatenate(chunks, axis=0)

    def init_params_split_keys(self, angle_key, normal_key):
        n_angle = sum(1 for s in self.specs if s.init == "angle")
        n_normal = sum(1 for s in self.specs if s.init == "normal")

        angle_keys = jax.random.split(angle_key, n_angle) if n_angle > 0 else None
        normal_keys = jax.random.split(normal_key, n_normal) if n_normal > 0 else None

        i_angle = 0
        i_normal = 0
        chunks = []
        for s in self.specs:
            if s.init == "angle":
                k = angle_keys[i_angle]
                i_angle += 1
                x = jax.random.uniform(k, shape=s.shape, minval=0.0, maxval=2.0 * jnp.pi)
            elif s.init == "normal":
                k = normal_keys[i_normal]
                i_normal += 1
                x = s.mean + s.scale * jax.random.normal(k, shape=s.shape)
            elif s.init == "fixed":
                x = jnp.full(s.shape, s.mean)
            else:
                x = jnp.zeros(s.shape)
            chunks.append(x.reshape(-1))
        return jnp.concatenate(chunks, axis=0)

    def init_params_su_ucr(self, su_key, ucr_key):
        n_su = sum(1 for s in self.specs if s.name.startswith("SU") or s.name.startswith("MTPLX"))
        n_ucr = sum(1 for s in self.specs if s.name.startswith("UCR"))

        su_keys = jax.random.split(su_key, n_su) if n_su > 0 else None
        ucr_keys = jax.random.split(ucr_key, n_ucr) if n_ucr > 0 else None

        i_su = 0
        i_ucr = 0
        chunks = []
        for s in self.specs:
            if s.name.startswith("SU") or s.name.startswith("MTPLX"):
                k = su_keys[i_su]
                i_su += 1
            elif s.name.startswith("UCR"):
                k = ucr_keys[i_ucr]
                i_ucr += 1
            else:
                k = su_keys[0] if su_keys is not None else ucr_keys[0]

            if s.init == "angle":
                x = jax.random.uniform(k, shape=s.shape, minval=0.0, maxval=2.0 * jnp.pi)
            elif s.init == "normal":
                x = s.mean + s.scale * jax.random.normal(k, shape=s.shape)
            elif s.init == "fixed":
                x = jnp.full(s.shape, s.mean)
            else:
                x = jnp.zeros(s.shape)
            chunks.append(x.reshape(-1))
        return jnp.concatenate(chunks, axis=0)

@jax.tree_util.register_pytree_node_class
class FullUcrVQSD:
    """Full-UCR VQSD model used as the Section 5 full-UCR baseline."""
    def __init__(self, n_anc: int, n_sys: int, su_depth: int | None = 2, mean_init: float | str = "pi/2", scale_init: float = 0.01):
        self.n_anc = int(n_anc)
        self.n_sys = int(n_sys)
        self.su_depth = su_depth if su_depth is None else int(su_depth)

        if isinstance(mean_init, str):
            if mean_init.lower() == "pi/2":
                mean_val = jnp.pi / 2
            else:
                mean_val = float(mean_init)
        else:
            mean_val = float(mean_init)

        su_dim, *_ = self.ansatz_specs(self.n_sys, depth=self.su_depth)
        self.su_param_dim = int(su_dim)

        specs = []
        fields = []

        specs.append(ParamSpec("SU_0", (self.su_param_dim,), init="normal", mean=jnp.pi/2, scale=0.01))
        fields.append(("SU_0", jnp.ndarray))

        specs.append(ParamSpec("UCR_0", (2 ** self.n_sys,), init="normal", mean=mean_val, scale=scale_init))
        fields.append(("UCR_0", jnp.ndarray))

        for i in range(1, self.n_anc):
            mt_name = f"MTPLX_{i}"
            specs.append(ParamSpec(mt_name, (2**i * self.su_param_dim,), init="normal", mean=jnp.pi/2, scale=0.01))
            fields.append((mt_name, jnp.ndarray))

            ucr_name = f"UCR_{i}"
            specs.append(ParamSpec(ucr_name, (2 ** (self.n_sys + i),), init="normal", mean=mean_val, scale=scale_init))
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(f"ParamsAncilla{self.n_anc}", fields, frozen=True)
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        self.layout = MetaParamLayout(specs, self.ParamsType)

        # ---- SU ansatz cache: wires(tuple) -> callable(params)->None ----
        self._ansatz_cache: Dict[Tuple[int, ...], Callable[[jnp.ndarray], None]] = {}

    # -----------------------------
    # public: layer application
    # -----------------------------
    def __call__(self, theta_flat, sys_wires, anc_wires):
        params = self.layout.unpack(theta_flat)
        self.block_layer(params, list(sys_wires), list(anc_wires))

    def block_layer(self, params, sys, anc):
        if len(anc) != self.n_anc:
            raise ValueError(f"Expected {self.n_anc} ancilla wires, got {len(anc)}.")

        # Base
        self.apply_su_ansatz(params.SU_0, sys)
        self.UCRy(params.UCR_0, ctrls=sys, target=anc[0])

        # Recursive
        for i in range(1, self.n_anc):
            mt_params = getattr(params, f"MTPLX_{i}")
            ucr_params = getattr(params, f"UCR_{i}")

            prev_anc = anc[:i]
            self.multiplexed_ansatz(mt_params, ctrls=prev_anc, targets=sys)

            ctrls_for_ucr = list(sys) + list(prev_anc)
            self.UCRy(ucr_params, ctrls=ctrls_for_ucr, target=anc[i])

    # -----------------------------
    # UCRy (multi-controlled RY)
    # -----------------------------
    def _mcry_on_target(self, angle, selector_vals, ctrls, target):
        if len(selector_vals) != len(ctrls):
            raise ValueError("selector_vals length must match ctrls.")
        qml.ctrl(qml.RY, control=ctrls, control_values=tuple(selector_vals))(angle, wires=target)


    def _apply_ucry(self, thetas, ctrls, target):
        # manual_ucry(thetas, ctrls, target) 만약 manual을 하고 싶다면 이런 식으로 아래 코드를 대체하면 됨
        k = len(ctrls)
        n_branches = 2 ** k

        if int(thetas.shape[0]) != n_branches:
            raise ValueError(f"UCRy expects {n_branches} parameters, got {thetas.shape[0]}")

        for idx in range(n_branches):
            angle = thetas[idx]
            selector_vals = tuple((idx >> (k - 1 - i)) & 1 for i in range(k))
            self._mcry_on_target(angle, selector_vals, ctrls, target)

    def UCRy(self, thetas, ctrls, target):
        self._apply_ucry(thetas, ctrls, target)

    # -----------------------------
    # Multiplexed SU ansatz (controlled by ancillas)
    # -----------------------------
    def _mcansatz_on_sys(self, params, selector_vals, ctrls, targets):
        if len(selector_vals) != len(ctrls):
            raise ValueError("selector_vals length must match ctrls.")
        qml.ctrl(
            self.apply_su_ansatz,
            control=ctrls,
            control_values=tuple(selector_vals),
        )(params, targets)

    def multiplexed_ansatz(self, thetas, ctrls, targets):
        k = len(ctrls)
        n_branches = 2 ** k
        dim = self.su_param_dim

        if int(thetas.shape[0]) != n_branches * dim:
            raise ValueError(
                f"MTPLX expects {n_branches*dim} parameters "
                f"({n_branches} branches * su_dim {dim}), got {thetas.shape[0]}"
            )

        for idx in range(n_branches):  # static python loop
            branch_params = thetas[idx * dim : (idx + 1) * dim]
            selector_vals = tuple((idx >> (k - 1 - i)) & 1 for i in range(k))
            self._mcansatz_on_sys(branch_params, selector_vals, ctrls, targets)

    # -----------------------------
    # SU(n) ansatz (ref-class faithful)
    # -----------------------------
    @staticmethod
    def ansatz_specs(n: int, depth: int | None = None):
        """
        Returns:
          su_dim, params_init, params_per_cz, num_cz, num_final_params

        - depth is None => "full" mode (4**n - 1) with partial final block allowed
        - depth is int  => cap number of CZ-blocks (each uses 4 params)
        """
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

        # depth-limited mode (depth can be 0)
        num_cz = int(max(0, min(int(depth), full_num_cz)))
        su_dim = params_init + num_cz * params_per_cz
        num_final_params = params_per_cz  # always 4 in fixed-depth repetition
        return int(su_dim), params_init, params_per_cz, num_cz, int(num_final_params)

    @staticmethod
    def _init_layer_su(params, wires):
        n = len(wires)
        for p, w in zip(params[:n], wires, strict=True):
            qml.RZ(p, wires=w)
        for p, w in zip(params[n : 2 * n], wires, strict=True):
            qml.RY(p, wires=w)
        for p, w in zip(params[2 * n : 3 * n], wires, strict=True):
            qml.RZ(p, wires=w)

    @staticmethod
    def _circuit_block_su(params, wires_2):
        qml.CZ(wires_2)
        qml.RY(params[0], wires=wires_2[0])
        qml.RY(params[1], wires=wires_2[1])
        qml.RZ(params[2], wires=wires_2[0])
        qml.RZ(params[3], wires=wires_2[1])

    @staticmethod
    def _final_circuit_block_su(params, wires_2, num_params: int):
        # partial last block for "full" mode
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
        n = len(wires)
        if n < 1:
            raise ValueError("The number of qubits n must be a positive integer.")

        su_dim, params_init, params_per_cz, num_cz, num_final_params = self.ansatz_specs(n, depth=self.su_depth)

        if n == 1:
            def ansatz(params):
                # params length == 3
                self._init_layer_su(params, wires)
            return ansatz

        # num_cz can be 0 (depth=0)
        if num_cz == 0:
            def ansatz(params):
                self._init_layer_su(params[:params_init], wires)
            return ansatz

        # block wiring schedule (same as ref)
        _block_wires = 2 * np.arange(num_cz) % (n - 1)
        if (n % 2) and num_cz > 0:
            _block_wires = _block_wires + (np.arange(num_cz) // (n // 2) % 2)
        _block_wires = _block_wires.astype(int)
        block_wires = [(wires[w], wires[w + 1]) for w in _block_wires.tolist()]

        # create slices: first (num_cz-1) blocks use 4 params, final uses num_final_params (1~4) in full-mode
        entries = []
        idx = params_init
        for i in range(num_cz):
            use_n = params_per_cz if (i < num_cz - 1) else int(num_final_params)
            sl = slice(idx, idx + use_n)
            entries.append((sl, block_wires[i], use_n))
            idx += use_n

        # sanity (especially for full-mode)
        if int(su_dim) != idx:
            # su_dim should match exactly the constructed parameter usage
            raise ValueError(f"Internal ansatz spec mismatch: su_dim={su_dim}, but consumed={idx}.")

        def ansatz(params):
            self._init_layer_su(params[:params_init], wires)
            for sl, w2, use_n in entries:
                if use_n == 4:
                    self._circuit_block_su(params[sl], w2)
                else:
                    self._final_circuit_block_su(params[sl], w2, use_n)

        return ansatz

    def _get_ansatz(self, wires):
        wires = tuple(wires)
        if wires not in self._ansatz_cache:
            self._ansatz_cache[wires] = self.make_ansatz(wires)
        return self._ansatz_cache[wires]

    def apply_su_ansatz(self, params, wires):
        ansatz_fn = self._get_ansatz(wires)
        ansatz_fn(params)

    # -----------------------------
    # pytree (static config only)
    # -----------------------------
    def tree_flatten(self):
        return (), (self.n_anc, self.n_sys, self.su_depth, 0.0)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_anc, n_sys, su_depth, mean_init = aux_data
        return cls(n_anc=n_anc, n_sys=n_sys, su_depth=su_depth, mean_init=mean_init)


@jax.tree_util.register_pytree_node_class
class RandomSparseFullUcrVQSD(FullUcrVQSD):
    """
    Full-UCR branch subset model.

    The SU and multiplexed-SU blocks match FullUcrVQSD, but each UCR block only
    instantiates the selected controlled-RY branches. This is equivalent to a
    full-UCR model with unselected UCR angles fixed at zero, without compiling
    those zero-angle controlled rotations.
    """

    def __init__(
        self,
        n_anc: int,
        n_sys: int,
        selected_ucr_indices: Sequence[Sequence[int]],
        su_depth: int | None = 2,
        mean_init: float | str = "pi/2",
        scale_init: float = 0.01,
    ):
        self.n_anc = int(n_anc)
        self.n_sys = int(n_sys)
        self.su_depth = su_depth if su_depth is None else int(su_depth)

        if isinstance(mean_init, str):
            if mean_init.lower() == "pi/2":
                mean_val = jnp.pi / 2
            else:
                mean_val = float(mean_init)
        else:
            mean_val = float(mean_init)

        if len(selected_ucr_indices) != self.n_anc:
            raise ValueError(
                f"Expected {self.n_anc} selected UCR index blocks, got {len(selected_ucr_indices)}."
            )
        selected_blocks: list[tuple[int, ...]] = []
        for block_idx, indices in enumerate(selected_ucr_indices):
            full_block_size = 2 ** (self.n_sys + block_idx)
            block = tuple(sorted(int(value) for value in indices))
            if len(set(block)) != len(block):
                raise ValueError(f"Duplicate selected UCR indices in block {block_idx}: {block}.")
            if block and (block[0] < 0 or block[-1] >= full_block_size):
                raise ValueError(
                    f"Selected UCR indices for block {block_idx} must be in "
                    f"[0, {full_block_size - 1}], got {block}."
                )
            selected_blocks.append(block)
        self.selected_ucr_indices = tuple(selected_blocks)

        su_dim, *_ = self.ansatz_specs(self.n_sys, depth=self.su_depth)
        self.su_param_dim = int(su_dim)

        specs = []
        fields = []

        specs.append(ParamSpec("SU_0", (self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
        fields.append(("SU_0", jnp.ndarray))

        specs.append(
            ParamSpec(
                "UCR_0",
                (len(self.selected_ucr_indices[0]),),
                init="normal",
                mean=mean_val,
                scale=scale_init,
            )
        )
        fields.append(("UCR_0", jnp.ndarray))

        for i in range(1, self.n_anc):
            mt_name = f"MTPLX_{i}"
            specs.append(ParamSpec(mt_name, (2**i * self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
            fields.append((mt_name, jnp.ndarray))

            ucr_name = f"UCR_{i}"
            specs.append(
                ParamSpec(
                    ucr_name,
                    (len(self.selected_ucr_indices[i]),),
                    init="normal",
                    mean=mean_val,
                    scale=scale_init,
                )
            )
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(f"ParamsAncilla{self.n_anc}RandomSparseUCR", fields, frozen=True)
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        self.layout = MetaParamLayout(specs, self.ParamsType)
        self._ansatz_cache: Dict[Tuple[int, ...], Callable[[jnp.ndarray], None]] = {}

    def block_layer(self, params, sys, anc):
        if len(anc) != self.n_anc:
            raise ValueError(f"Expected {self.n_anc} ancilla wires, got {len(anc)}.")

        self.apply_su_ansatz(params.SU_0, sys)
        self.sparse_UCRy(
            params.UCR_0,
            selected_indices=self.selected_ucr_indices[0],
            ctrls=sys,
            target=anc[0],
        )

        for i in range(1, self.n_anc):
            mt_params = getattr(params, f"MTPLX_{i}")
            ucr_params = getattr(params, f"UCR_{i}")
            prev_anc = anc[:i]
            self.multiplexed_ansatz(mt_params, ctrls=prev_anc, targets=sys)

            ctrls_for_ucr = list(sys) + list(prev_anc)
            self.sparse_UCRy(
                ucr_params,
                selected_indices=self.selected_ucr_indices[i],
                ctrls=ctrls_for_ucr,
                target=anc[i],
            )

    def sparse_UCRy(self, thetas, selected_indices: Sequence[int], ctrls, target):
        if int(thetas.shape[0]) != len(selected_indices):
            raise ValueError(
                f"RandomSparseFullUcrVQSD sparse_UCRy expects {len(selected_indices)} "
                f"parameters, got {thetas.shape[0]}."
            )
        k = len(ctrls)
        for theta_idx, full_branch_idx in enumerate(selected_indices):
            selector_vals = tuple((int(full_branch_idx) >> (k - 1 - bit)) & 1 for bit in range(k))
            self._mcry_on_target(thetas[theta_idx], selector_vals, ctrls, target)

    def tree_flatten(self):
        return (), (self.n_anc, self.n_sys, self.selected_ucr_indices, self.su_depth, 0.0)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_anc, n_sys, selected_ucr_indices, su_depth, mean_init = aux_data
        return cls(
            n_anc=n_anc,
            n_sys=n_sys,
            selected_ucr_indices=selected_ucr_indices,
            su_depth=su_depth,
            mean_init=mean_init,
        )


@jax.tree_util.register_pytree_node_class
class WalshKLocalVQSD(FullUcrVQSD):
    """
    Section 5 Walsh degree-k UCR model.

    Each UCR block keeps a degree-0 bias RY and adds one Walsh-parity RY
    parameter for every nonempty control subset with size <= `ucr_degree`.
    Subsets are ordered by increasing degree and then by
    `itertools.combinations(ctrls, local_degree)`.
    """

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
        self.n_anc = int(n_anc)
        self.n_sys = int(n_sys)
        self.su_depth = su_depth if su_depth is None else int(su_depth)
        self.ucr_degree = int(ucr_degree)
        if self.ucr_degree < 1:
            raise ValueError(f"ucr_degree must be >= 1, got {ucr_degree}")

        if isinstance(mean_init, str):
            if mean_init.lower() == "pi/2":
                mean_val = jnp.pi / 2
            else:
                mean_val = float(mean_init)
        else:
            mean_val = float(mean_init)

        if isinstance(bias_mean_init, str):
            if bias_mean_init.lower() == "pi/2":
                bias_mean_val = jnp.pi / 2
            elif bias_mean_init.lower() == "pi":
                bias_mean_val = jnp.pi
            else:
                bias_mean_val = float(bias_mean_init)
        else:
            bias_mean_val = float(bias_mean_init)

        self.mean_init = float(mean_val)
        self.bias_mean_init = float(bias_mean_val)
        self.scale_init = float(scale_init)
        self.bias_scale_init = float(bias_scale_init)

        su_dim, *_ = self.ansatz_specs(self.n_sys, depth=self.su_depth)
        self.su_param_dim = int(su_dim)

        specs = []
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

        for i in range(1, self.n_anc):
            mt_name = f"MTPLX_{i}"
            specs.append(ParamSpec(mt_name, (2**i * self.su_param_dim,), init="normal", mean=jnp.pi / 2, scale=0.01))
            fields.append((mt_name, jnp.ndarray))

            ucr_bias_name = f"UCR_BIAS_{i}"
            specs.append(ParamSpec(ucr_bias_name, (1,), init="normal", mean=self.bias_mean_init, scale=self.bias_scale_init))
            fields.append((ucr_bias_name, jnp.ndarray))

            ucr_name = f"UCR_{i}"
            specs.append(
                ParamSpec(
                    ucr_name,
                    (self.num_k_local_terms(self.n_sys + i, self.ucr_degree),),
                    init="normal",
                    mean=self.mean_init,
                    scale=self.scale_init,
                )
            )
            fields.append((ucr_name, jnp.ndarray))

        self.ParamsType = make_dataclass(
            f"ParamsAncilla{self.n_anc}WalshDegree{self.ucr_degree}",
            fields,
            frozen=True,
        )
        jax.tree_util.register_dataclass(
            self.ParamsType,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        self.layout = MetaParamLayout(specs, self.ParamsType)
        self._ansatz_cache: Dict[Tuple[int, ...], Callable[[jnp.ndarray], None]] = {}

    @staticmethod
    def num_k_local_terms(num_controls: int, degree: int) -> int:
        k = int(num_controls)
        d = int(degree)
        if k < 0:
            raise ValueError(f"num_controls must be >= 0, got {num_controls}")
        if d < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        return int(sum(math.comb(k, r) for r in range(1, min(d, k) + 1)))

    def block_layer(self, params, sys, anc):
        if len(anc) != self.n_anc:
            raise ValueError(f"Expected {self.n_anc} ancilla wires, got {len(anc)}.")

        self.apply_su_ansatz(params.SU_0, sys)
        qml.RY(params.UCR_BIAS_0[0], wires=anc[0])
        self.UCRy(params.UCR_0, ctrls=sys, target=anc[0])

        for i in range(1, self.n_anc):
            mt_params = getattr(params, f"MTPLX_{i}")
            ucr_bias = getattr(params, f"UCR_BIAS_{i}")
            ucr_params = getattr(params, f"UCR_{i}")

            prev_anc = anc[:i]
            self.multiplexed_ansatz(mt_params, ctrls=prev_anc, targets=sys)

            qml.RY(ucr_bias[0], wires=anc[i])
            ctrls_for_ucr = list(sys) + list(prev_anc)
            self.UCRy(ucr_params, ctrls=ctrls_for_ucr, target=anc[i])

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
            raise ValueError(f"WalshKLocalVQSD.UCRy expects {expected} parameters, got {len(thetas)}")

        theta_idx = 0
        max_degree = min(self.ucr_degree, len(ctrls))
        for local_degree in range(1, max_degree + 1):
            for ctrl_group in combinations(ctrls, local_degree):
                self._walsh_parity_ry(thetas[theta_idx], ctrls=list(ctrl_group), target=target)
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
