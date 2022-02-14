import numpy as np
from .parameters import params, x_idx


def get_obj(n_controlled, n_inter):
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    f = np.zeros((params.N, n_var))  # linear term
    H = np.zeros((params.N, n_var, n_var))# Hessian
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            ddS_idx = (params.n_binputs + params.n_slacks) * n_inter + (
                    params.n_cinputs + params.n_states) * p_idx + x_idx.ddS
            H[stage_idx, ddS_idx, ddS_idx] = params.penalty_input
            if stage_idx == params.N - 1:
                S_idx = (params.n_binputs + params.n_slacks) * n_inter + (
                        params.n_cinputs + params.n_states) * p_idx + x_idx.S
                f[stage_idx, S_idx] = -params.s_reward  # only get reward at terminal stage
    return f, H
