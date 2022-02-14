import numpy as np
from .parameters import params


def get_eq(n_controlled, n_inter):
    # C_{i-1}z_{i-1} + D_{i}z_{i} = c_{i}
    # z=[(ub, s) for all intersections, (uc, x)for all controlled players]
    # eq: [auxiliary cons for each intersection, dynamics for each controlled player]
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    n_eq = params.n_binputs * n_inter + params.n_states * n_controlled
    C = np.zeros((params.N, n_eq, n_var))
    D = np.zeros((params.N, n_eq, n_var))
    c = np.zeros((params.N, n_eq, 1))
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            eq_start_idx = params.n_binputs * n_inter + p_idx*params.n_states
            eq_idx = list(range(eq_start_idx, eq_start_idx+params.n_states))
            z_start_idx = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx
            z_idx = list(range(z_start_idx, z_start_idx + params.n_cinputs + params.n_states))
            if stage_idx < params.N - 1:
                C_x = params.C
                C[stage_idx, eq_idx, z_idx] = C_x
            if stage_idx > 0:
                c_x = np.zeros((params.nx, 1))
                c[stage_idx, eq_idx] = c_x
                D_x = np.hstack((np.zeros((params.n_states, params.n_cinputs)), -np.eye(params.n_states)))
                D[stage_idx, eq_idx, z_idx] = D_x
            if stage_idx == 0:
                D[stage_idx, eq_idx, z_idx] = np.hstack((np.zeros((params.n_states, params.n_cinputs)), -np.eye(params.n_states)))
        for i_idx in range(n_inter):
            eq_start_idx = params.n_binputs * i_idx
            eq_idx = list(range(eq_start_idx, eq_start_idx + params.n_binputs))
            z_start_idx = (params.n_binputs + params.n_slacks) * i_idx
            z_idx = list(range(z_start_idx, z_start_idx + params.n_binputs + params.n_slacks))
            if stage_idx < params.N - 1:
                C_sigma = np.hstack((-np.eye(params.n_binputs), np.zeros((params.n_binputs, params.n_slacks))))
                C[stage_idx, eq_idx, z_idx] = C_sigma
            if stage_idx > 0:
                c_sigma = np.zeros((params.n_binputs, 1))
                c[stage_idx, eq_idx] = c_sigma
                D_sigma = np.hstack((np.eye(params.n_binputs), np.diag(np.array([1, -1, 1, -1]))))
                D[stage_idx, eq_idx, z_idx] = D_sigma
    return n_eq, C, D, c
