import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from geometry import SE2value
from dg_commons import PlayerName, DgSampledSequence
from homotopies.MIQP.utils.intersects import get_box
from .parameters import params, x_idx, ub_idx, player_idx
from ..homotopy.homotopy import Homotopy


def get_bounds(n_controlled: int, n_inter: int):
    bin_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    slack_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    inter_bounds = np.tile(np.vstack((bin_bounds, slack_bounds)), (n_inter, 1))
    acc_bounds = np.array(params.vehicle_params.acc_limits)
    v_bounds = np.array(params.vehicle_params.vx_limits)
    default_bounds = np.array([0, params.M])
    player_bounds = np.tile(np.vstack((acc_bounds, default_bounds, v_bounds)), (n_controlled, 1))
    return np.vstack((inter_bounds, player_bounds))


def get_ineq(
    n_controlled: int,
    n_inter: int,
    trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
    intersects: Dict[PlayerName, Dict[PlayerName, float]],
    homotopies: Optional[Homotopy],
    box_buffer: float = 1.0,
    use_homo=True,
):
    """
    This function returns matrices A and b for polytopic inequality constraints:
        A_{i}z_{i} <= b_{i}
    variable order: z=[(ub, s) for all intersections, (uc, x)for all controlled players]
    intersections order: (1,2), (1,3),..,(1, n_player), (2,3),..,(2, n_player)
    binary variable order: sigma1_n, sigma1_p, sigma2_n, sigma2_p
    ineq order: [(homotopy cons, box cons) for each intersection]
    """
    assert n_inter == int(n_controlled * (n_controlled - 1) / 2)
    if use_homo:
        n_ineq = params.n_ineq
    else:
        n_ineq = params.n_ineq - 1  # 2 homotopy constrants become 1 collision avoidance constraint
    n_ineqs = n_ineq * n_inter
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    A = np.zeros((n_ineqs, n_var))
    b = np.zeros((n_ineqs, 1))
    for i_idx, player_pair in enumerate(combinations(trajs.keys(), 2)):
        player1 = player_pair[0]
        player2 = player_pair[1]
        ineq_idx = list(range(n_ineq * i_idx, n_ineq * (i_idx + 1)))
        ub_start_idx = (params.n_binputs + params.n_slacks) * i_idx
        x1_idx = (
            (params.n_binputs + params.n_slacks) * n_inter
            + (params.n_cinputs + params.n_states) * player_idx[player1]
            + x_idx.S
        )
        x2_idx = (
            (params.n_binputs + params.n_slacks) * n_inter
            + (params.n_cinputs + params.n_states) * player_idx[player2]
            + x_idx.S
        )
        z_idx = list(range(ub_start_idx, ub_start_idx + params.n_binputs)) + [x1_idx] + [x2_idx]

        if use_homo:
            temp_A = A[ineq_idx, :]
            temp_A[:, z_idx] = get_ineq_A()
            A[ineq_idx, :] = temp_A

            box = get_box(trajs, intersects, player1, player2, box_buffer)
            if player2 in intersects[player1].keys():
                homo_class = homotopies.homo_class
                h = homo_class[player1][player2]
            else:  # player1 and player2 don't intersect, use default homotopy class h=0
                h = 0
            b[ineq_idx, :] = get_ineq_b(box, h)
        else:
            temp_A = A[ineq_idx, :]
            temp_A[:, z_idx] = get_ineq_A_no_homo()
            A[ineq_idx, :] = temp_A

            box = get_box(trajs, intersects, player1, player2, box_buffer)
            b[ineq_idx, :] = get_ineq_b_no_homo(box)
    return A, b


def get_ineq_A():
    """
    this function constructs the A matrix of all inequality constraints for one intersection
    ineq order: [homotopy cons, box cons]
    z slice : [binary variables, s1, s2]
    """
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
    """
    this function constructs the b vector of all inequality constraints for one intersection
    ineq order: [homotopy cons, box cons]
    z slice: [binary variables, s1, s2]
    """
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    b = np.zeros((params.n_ineq, 1))
    b[0] = -h
    b[1] = h - 1

    b[2] = params.M + center_x - w_x / 2
    b[3] = -(center_x - w_x / 2)

    b[4] = -(center_x + w_x / 2) + params.M
    b[5] = center_x + w_x / 2

    b[6] = params.M + center_y - w_y / 2
    b[7] = -(center_y - w_y / 2)

    b[8] = -(center_y + w_y / 2) + params.M
    b[9] = center_y + w_y / 2

    return b


def get_ineq_A_no_homo():
    """
    this function constructs the A matrix of all inequality constraints for one intersection
    ineq order: [homotopy cons, box cons]
    z slice : [binary variables, s1, s2]
    """
    A = np.zeros(((params.n_ineq - 1), 6))
    s1_idx = params.n_binputs
    s2_idx = params.n_binputs + 1
    A[0, [ub_idx.Sigma1_n, ub_idx.Sigma1_p, ub_idx.Sigma2_n, ub_idx.Sigma2_p]] = -1

    A[1, ub_idx.Sigma1_n] = params.M
    A[1, s1_idx] = 1
    A[2, ub_idx.Sigma1_n] = -params.M
    A[2, s1_idx] = -1

    A[3, ub_idx.Sigma1_p] = params.M
    A[3, s1_idx] = -1
    A[4, ub_idx.Sigma1_p] = -params.M
    A[4, s1_idx] = 1

    A[5, ub_idx.Sigma2_n] = params.M
    A[5, s2_idx] = 1
    A[6, ub_idx.Sigma2_n] = -params.M
    A[6, s2_idx] = -1

    A[7, ub_idx.Sigma2_p] = params.M
    A[7, s2_idx] = -1
    A[8, ub_idx.Sigma2_p] = -params.M
    A[8, s2_idx] = 1
    return A


def get_ineq_b_no_homo(box):
    """
    this function constructs the b vector of all inequality constraints for one intersection
    ineq order: [homotopy cons, box cons]
    z slice: [binary variables, s1, s2]
    """
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    b = np.zeros(((params.n_ineq - 1), 1))
    b[0] = -1

    b[1] = params.M + center_x - w_x / 2
    b[2] = -(center_x - w_x / 2)

    b[3] = -(center_x + w_x / 2) + params.M
    b[4] = center_x + w_x / 2

    b[5] = params.M + center_y - w_y / 2
    b[6] = -(center_y - w_y / 2)

    b[7] = -(center_y + w_y / 2) + params.M
    b[8] = center_y + w_y / 2

    return b
