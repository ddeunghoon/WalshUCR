from __future__ import annotations

from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import pennylane as qml


WiresLike = Union[int, Sequence[int]]


def _as_wires_list(wires: WiresLike) -> list[int]:
    if isinstance(wires, int):
        return [wires]
    return list(wires)


def _infer_batch_size_weyl(inputs: Tuple[Any, Any]) -> int:
    a, _ = inputs
    return int(np.asarray(a).shape[0])


def _prep_fns_to_density(
    prep_fns: list[Callable[[WiresLike], None]],
    wires: WiresLike,
    n_wires_total: int,
    interface: str = "numpy",
) -> list[Any]:
    wires_list = _as_wires_list(wires)
    dev = qml.device("default.mixed", wires=n_wires_total, shots=None)

    @qml.qnode(dev, interface=interface)
    def _qnode(prep_fn: Callable[[WiresLike], None]):
        prep_fn(wires_list)
        return qml.density_matrix(wires_list)

    return [_qnode(prep_fn) for prep_fn in prep_fns]


def make_rhos(
    benchmark_type: str,
    inputs: Any,
    circuit_fn: Callable[[Any, int], None],
    n_sys: int,
    sys_wires: WiresLike,
    interface: str = "numpy",
) -> list[Any]:
    bt = benchmark_type.lower().strip()
    if bt != "weyl":
        raise ValueError(f"Unsupported benchmark type for WalshUCR: {benchmark_type}. Expected 'weyl'.")

    wires_list = _as_wires_list(sys_wires)
    n_wires_total = max(n_sys, (max(wires_list) + 1) if wires_list else n_sys)
    if not (isinstance(inputs, (tuple, list)) and len(inputs) == 2):
        raise ValueError("[weyl] inputs must be a tuple/list (a_batch, b_batch).")

    a_batch, b_batch = inputs
    batch_size = _infer_batch_size_weyl((a_batch, b_batch))
    prep_fns: list[Callable[[WiresLike], None]] = []
    for idx in range(batch_size):
        a_i = a_batch[idx]
        b_i = b_batch[idx]

        def _prep(_wires, a_i=a_i, b_i=b_i):
            circuit_fn((a_i, b_i), n_sys)

        prep_fns.append(_prep)

    return _prep_fns_to_density(
        prep_fns=prep_fns,
        wires=wires_list,
        n_wires_total=n_wires_total,
        interface=interface,
    )
