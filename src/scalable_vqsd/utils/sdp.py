from __future__ import annotations

import cvxpy as cp
from pennylane import numpy as np


def sdp_med(q_rho, num_state, num_povm=3):
    dim = q_rho[0].shape[0]
    effects = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_povm)]
    constraints = [sum(effects) == np.eye(dim)]
    constraints += [effect >> 0 for effect in effects]

    objective = 1
    for idx in range(num_state):
        objective -= cp.real(cp.trace(effects[idx] @ q_rho[idx]))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return problem.value, [effect.value for effect in effects]
