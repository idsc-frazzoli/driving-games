import numpy as np
from .parameters import params, x_idx, ub_idx, player_idx
from homotopies.MILP.utils.intersects import get_box
from typing import Dict, List, Tuple, Optional
from dg_commons import PlayerName, DgSampledSequence
from geometry import SE2value
from itertools import combinations


def get_bounds(n_controlled,
               n_inter):
    bin_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    slack_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    inter_bounds = np.tile(np.vstack((bin_bounds, slack_bounds)), (n_inter, 1))
    acc_bounds = np.array(params.vehicle_params.acc_limits)
    v_bounds = np.array(params.vehicle_params.vx_limits)
    default_bounds = np.array([0, params.M])
    player_bounds = np.tile(np.vstack((acc_bounds,
                                       default_bounds,
                                       v_bounds)),
                            (n_controlled, 1))
    return np.vstack((inter_bounds, player_bounds))


def get_ineq(n_controlled,
             n_inter,
             trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
             intersects: Dict[PlayerName, Dict[PlayerName, float]],
             homotopies: Dict[PlayerName, Dict[PlayerName, int]],
             box_buffer: float=1.):
    # A_{i}z_{i} <= b_{i}
    # z=[(ub, s) for all intersections, (uc, x)for all controlled players]
    # order of intersections: (1,2), (1,3),..,(1, n_player), (2,3),..,(2, n_player)
    # order of ub: sigma1_n, sigma1_p, sigma2_n, sigma2_p
    # ineq: [homotopy and box cons for each intersection]
    assert n_inter == int(n_controlled * (n_controlled - 1) / 2)
    n_ineqs = params.n_ineq * n_inter
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    A = np.zeros((n_ineqs, n_var))
    b = np.zeros((n_ineqs, 1))
    i_idx = 0
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        ineq_idx = list(range(params.n_ineq * i_idx, params.n_ineq * (i_idx + 1)))
        ub_start_idx = (params.n_binputs + params.n_slacks) * i_idx
        i_idx += 1
        x1_idx = (params.n_binputs + params.n_slacks) * n_inter + \
                 (params.n_cinputs + params.n_states) * player_idx[player1] + \
                 x_idx.S
        x2_idx = (params.n_binputs + params.n_slacks) * n_inter + \
                 (params.n_cinputs + params.n_states) * player_idx[player2] + \
                 x_idx.S
        z_idx = list(range(ub_start_idx, ub_start_idx + params.n_binputs)) + [x1_idx] + [x2_idx]
        box = get_box(trajs, intersects, player1, player2, box_buffer)
        temp_A = A[ineq_idx, :]
        temp_A[:, z_idx] = get_ineq_A(box)
        A[ineq_idx, :] = temp_A
        h = homotopies[player1][player2]
        b[ineq_idx, :] = get_ineq_b(box, h)
    return A, b


def get_ineq_A(box):
    #  box = ((center_x, center_y), w_x, w_y)
    #  z=[binary variables, s1, s2]
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    A = np.zeros((params.n_ineq, 6))
    s1_idx = params.n_binputs
    s2_idx = params.n_binputs + 1
    A[0, [ub_idx.Sigma1_n, ub_idx.Sigma2_p]] = -1
    A[1, [ub_idx.Sigma1_p, ub_idx.Sigma2_n]] = -1

    A[2, ub_idx.Sigma1_n] = params.M
    A[2, s1_idx] = 1
    A[3, ub_idx.Sigma1_n] = -params.M
    A[3, s1_idx] = -1

    A[4, ub_idx.Sigma1_p] = params.M
    A[4, s1_idx] = -1
    A[5, ub_idx.Sigma1_p] = -params.M
    A[5, s1_idx] = 1

    A[6, ub_idx.Sigma2_n] = params.M
    A[6, s2_idx] = 1
    A[7, ub_idx.Sigma2_n] = -params.M
    A[7, s2_idx] = -1

    A[8, ub_idx.Sigma2_p] = params.M
    A[8, s2_idx] = -1
    A[9, ub_idx.Sigma2_p] = -params.M
    A[9, s2_idx] = 1
    return A


def get_ineq_b(box, h):
    #  box = ((center_x, center_y), w_x, w_y)
    #  z=[binary variables, s1, s2]
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    b = np.zeros((params.n_ineq, 1))
    b[0] = -h
    b[1] = h - 1

    b[2] = params.M + center_x - w_x / 2
    b[3] = - (center_x - w_x / 2)

    b[4] = -(center_x + w_x / 2) + params.M
    b[5] = center_x + w_x / 2

    b[6] = params.M + center_y - w_y / 2
    b[7] = - (center_y - w_y / 2)

    b[8] = -(center_y + w_y / 2) + params.M
    b[9] = center_y + w_y / 2

    return b
