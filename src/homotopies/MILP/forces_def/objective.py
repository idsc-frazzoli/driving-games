import numpy as np
from .parameters import params, x_idx, uc_idx


def get_obj(n_controlled, n_inter):
    """
    cost = sum_{i=0}^{N-1} p*|dds_{i}|^2 + r*(s_{N-1}-s_{0})
    """
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    f = np.zeros((params.N, n_var))  # linear term
    H = np.zeros((params.N, n_var, n_var))# Hessian
    for p_idx in range(n_controlled):
        for stage_idx in range(params.N):
            ddS_idx = (params.n_binputs + params.n_slacks) * n_inter + (
                    params.n_cinputs + params.n_states) * p_idx + uc_idx.ddS
            H[stage_idx, ddS_idx, ddS_idx] = params.penalty_input
        S_idx = (params.n_binputs + params.n_slacks) * n_inter + (
                params.n_cinputs + params.n_states) * p_idx + x_idx.S
        f[params.N - 1, S_idx] = -params.s_reward  # get reward on the progress made in N stages
        f[0, S_idx] = params.s_reward
    return f, H
