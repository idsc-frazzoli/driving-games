import numpy as np
from .parameters import params, x_idx, ub_idx
from homotopies.MILP.utils.intersects import get_box
from typing import Dict, List, Tuple, Optional
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from geometry import SE2value
from itertools import combinations

player_idx = {'p1': 0, 'p2': 1, 'p3': 2}
controlled_players = ['p1', 'p2', 'p3']
vehicle_params = VehicleParameters.default_car()


def get_bounds(n_controlled,
               n_inter):
    bin_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    slack_bounds = np.hstack((np.zeros((params.n_binputs, 1)), np.ones((params.n_binputs, 1))))
    inter_bounds = np.tile(np.vstack((bin_bounds, slack_bounds)), (n_inter, 1))
    acc_bounds = np.array(vehicle_params.acc_limits)
    v_bounds = np.array(vehicle_params.vx_limits)
    default_bounds = np.array([0, params.M])
    car_bounds = np.tile(np.vstack((acc_bounds,
                                    default_bounds,
                                    v_bounds)),
                         (n_controlled, 1))
    return np.vstack((inter_bounds, car_bounds))


def get_ineq(n_controlled,
             n_inter,
             trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
             intersects: Dict[PlayerName, Dict[PlayerName, float]],
             homotopies: Dict[PlayerName, Dict[PlayerName, int]]):
    # A_{i}z_{i} <= b_{i}
    # z=[(ub, s) for all intersections, (uc, x)for all controlled players]
    # order of intersections: (1,2), (1,3),..,(1, n_player), (2,3),..,(2, n_player)
    # order of ub: sigma1_n, sigma1_p, sigma2_n, sigma2_p
    # ineq: [homotopy and box cons for each intersection]
    assert n_inter == int(n_controlled*(n_controlled-1)/2)
    n_ineqs = params.n_ineq * n_inter
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    A = np.zeros((n_ineqs, n_var))
    b = np.zeros(n_ineqs, 1)
    i_idx = 0
    for player_pair in combinations(player_idx.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        ineq_idx = list(range(params.n_ineq * i_idx, params.n_ineq * (i_idx + 1)))
        ub_start_idx = (params.n_binputs + params.n_slacks) * i_idx
        x1_idx = (params.n_binputs + params.n_slacks) * n_inter + \
                 (params.n_cinputs + params.n_states) * player_idx[player1] + \
                 x_idx.S
        x2_idx = (params.n_binputs + params.n_slacks) * n_inter + \
                 (params.n_cinputs + params.n_states) * player_idx[player2] + \
                 x_idx.S
        z_idx = list(range(ub_start_idx, ub_start_idx + params.n_binputs)) + [x1_idx] + [x2_idx]
        box = get_box(trajs, intersects, player1, player2)
        A[ineq_idx, z_idx] = get_ineq_A(box)
        h = homotopies[player1][player2]
        b[ineq_idx, :] = get_ineq_b(box, h)
    return A, b


def get_ineq_A(box):
    #  box = ((center_x, center_y), w_x, w_y)
    #  z=[binary variables, s1, s2]
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    A = np.zeros((params.nineq, 6))
    s1_idx = params.n_binputs
    s2_idx = params.n_binputs + 1
    A[0, [ub_idx.Sigma1_n, ub_idx.Sigma1_p]] = -1
    A[1, [ub_idx.Sigma1_p, ub_idx.Sigma2_n]] = -1
    A[2, ub_idx.Sigma1_n] = params.M
    A[2, s1_idx] = 1
    A[3, ub_idx.Sigma1_p] = center_x + w_x / 2
    A[3, s1_idx] = -1
    A[4, ub_idx.Sigma2_n] = params.M
    A[4, s2_idx] = 1
    A[3, ub_idx.Sigma2_p] = center_y + w_y / 2
    A[3, s2_idx] = -1
    return A


def get_ineq_b(box, h):
    #  box = ((center_x, center_y), w_x, w_y)
    #  z=[binary variables, s1, s2]
    center_x, center_y = box[0]
    w_x = box[1]
    w_y = box[2]
    b = np.zeros((params.nineq, 1))
    b[0] = -1 + h
    b[1] = -h
    b[2] = params.M + center_x - w_x / 2
    b[3] = 0
    b[4] = center_y - w_y / 2 + params.M
    b[5] = 0

    return b
