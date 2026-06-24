from __future__ import annotations

import jax
import pennylane as qml


def make_batched_qnode(
    *,
    n_sys: int,
    sys_wires,
    anc_wires,
    all_wires,
    model,
    benchmark,
    device_name: str = "default.qubit",
    diff_method: str = "backprop",
):
    dev = qml.device(device_name, wires=all_wires)
    circuit_fn = benchmark.get_circuit_fn()
    vmap_axes_inputs = benchmark.vmap_axes

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(inputs, params):
        circuit_fn(inputs, n_sys)
        model(params, sys_wires, anc_wires)
        return qml.probs(wires=anc_wires)

    return jax.vmap(circuit, in_axes=(vmap_axes_inputs, None))
