import numpy as np
from .parameters import params, x_idx, uc_idx


def get_obj(n_controlled, n_inter):
    """
    cost = sum_{i=0}^{N-1} p*|dds_{i}|^2 + r*(s_{N-1}-s_{0})
    """
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    f = np.zeros((params.N, n_var))  # linear term
    H = np.zeros((params.N, n_var, n_var))  # Hessian
    for p_idx in range(n_controlled):
        for stage_idx in range(params.N):
            ddS_idx = (
                (params.n_binputs + params.n_slacks) * n_inter
                + (params.n_cinputs + params.n_states) * p_idx
                + uc_idx.ddS
            )
            H[stage_idx, ddS_idx, ddS_idx] = params.penalty_input
        S_idx = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx + x_idx.S
        f[params.N - 1, S_idx] = -params.s_reward  # get reward on the progress made in N stages
        f[0, S_idx] = params.s_reward
    return f, H


def get_obj_velocity(n_controlled, n_inter, target_velocity):
    """
    ATTENTION: this cost function is imcomplete because low-level interface doesn't support 2-norm of (ds-ds_target)
    cost = sum_{i=0}^{N-1} p1*|dds_{i}|^2 + p2*|ds_{i}-ds_target_{i}|^2
    target_velocity: n_controlled length array [v1, v2, v3]
    """
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    f = np.zeros((params.N, n_var))  # linear term
    H = np.zeros((params.N, n_var, n_var))  # Hessian
    for p_idx in range(n_controlled):
        for stage_idx in range(params.N):
            ddS_idx = (
                (params.n_binputs + params.n_slacks) * n_inter
                + (params.n_cinputs + params.n_states) * p_idx
                + uc_idx.ddS
            )
            H[stage_idx, ddS_idx, ddS_idx] = params.penalty_input
            dS_idx = (
                (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx + x_idx.dS
            )
            H[stage_idx, dS_idx, dS_idx] = params.penalty_velocity
        S_idx = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx + x_idx.S
        f[params.N - 1, S_idx] = 0  # get reward on the progress made in N stages
        f[0, S_idx] = 0
    return f, H
