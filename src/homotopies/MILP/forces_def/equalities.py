import numpy as np
from itertools import combinations
from homotopies.MILP.utils.intersects import get_box
from .parameters import params, player_idx, x_idx, ub_idx


def get_eq(n_controlled, n_inter, use_bin_init=False):
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

            C_temp = np.zeros_like(C[stage_idx, eq_idx, :])
            C_temp[:, z_idx] = params.C
            C[stage_idx, eq_idx, :] = C_temp

            c_x = np.zeros((params.n_states, 1))
            c[stage_idx, eq_idx] = c_x

            D_x = np.hstack((np.zeros((params.n_states, params.n_cinputs)), -np.eye(params.n_states)))
            D_temp = np.zeros_like(D[stage_idx, eq_idx, :])
            D_temp[:, z_idx] = D_x
            D[stage_idx, eq_idx, :] = D_temp

        for i_idx in range(n_inter):
            eq_start_idx = params.n_binputs * i_idx
            eq_idx = list(range(eq_start_idx, eq_start_idx + params.n_binputs))
            z_start_idx = (params.n_binputs + params.n_slacks) * i_idx
            z_idx = list(range(z_start_idx, z_start_idx + params.n_binputs + params.n_slacks))

            C_sigma = np.hstack((-np.eye(params.n_binputs), np.zeros((params.n_binputs, params.n_slacks))))
            C_temp = np.zeros_like(C[stage_idx, eq_idx, :])
            C_temp[:, z_idx] = C_sigma
            C[stage_idx, eq_idx, :] = C_temp

            c_sigma = np.zeros((params.n_binputs, 1))
            c[stage_idx, eq_idx] = c_sigma

            if stage_idx > 0:
                D_sigma = np.hstack((np.eye(params.n_binputs), np.diag(np.array([1, -1, 1, -1]))))
                D_temp = np.zeros_like(D[stage_idx, eq_idx, :])
                D_temp[:, z_idx] = D_sigma
                D[stage_idx, eq_idx, :] = D_temp
            if stage_idx == 0:
                D_temp = np.zeros_like(D[stage_idx, eq_idx, :])
                D_temp[:, z_idx] = np.hstack((-np.eye(params.n_binputs), np.zeros((params.n_binputs, params.n_slacks))))
                D[stage_idx, eq_idx, :] = D_temp

    return n_eq, C, D, c


def get_init_bin(X_curr, trajs, intersects, n_inter, box_buffer):
    init_bin = np.zeros(params.n_binputs * n_inter)
    i_idx = 0
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        s1_idx = player_idx[player1] * params.n_states + x_idx.S - params.n_cinputs
        s2_idx = player_idx[player2] * params.n_states + x_idx.S - params.n_cinputs
        s1 = X_curr[s1_idx]
        s2 = X_curr[s2_idx]
        (center1, center2), w1, w2 = get_box(trajs, intersects, player1, player2, box_buffer)
        if s1 < center1 - w1/2:
            init_bin[i_idx*params.n_binputs+ub_idx.Sigma1_n] = 1
        elif s1 > center1 + w1/2:
            init_bin[i_idx*params.n_binputs+ub_idx.Sigma1_p] = 1
        if s2 < center2 - w2/2:
            init_bin[i_idx*params.n_binputs+ub_idx.Sigma2_n] = 1
        elif s2 > center2 + w2/2:
            init_bin[i_idx*params.n_binputs+ub_idx.Sigma2_p] = 1
        i_idx += 1
    return init_bin
