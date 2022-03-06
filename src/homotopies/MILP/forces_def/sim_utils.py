import numpy as np
from itertools import combinations
from homotopies.MILP.utils.intersects import get_box
from .parameters import params, player_idx, x_idx, ub_idx, uc_idx

__all__ = ["get_init_bin",
           "update_sim",
           "extract_plans",
           "check_status",
           "update_evaluation"]


def get_init_bin(X_curr, trajs, intersects, n_inter, box_buffer):
    """compute the value of the binary variables at stage 0, which is deterministic given current states"""
    init_bin = np.zeros(params.n_binputs * n_inter)
    for i_idx, player_pair in enumerate(combinations(trajs.keys(), 2)):
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
    return init_bin


def update_sim(X, ddS, n_controlled):
    """
    update simulator, assuming the simulation model is the same as the control model
    """
    next_X = np.zeros_like(X)
    for p_idx in range(n_controlled):
        x_idx = list(range(p_idx * params.n_states, (p_idx + 1) * params.n_states))
        dds_idx = list(range(p_idx * params.n_cinputs, (p_idx + 1) * params.n_cinputs))
        curr = np.hstack((ddS[dds_idx], X[x_idx])).reshape([-1, 1])
        next_X[x_idx] = np.dot(params.C, curr).reshape([2, ])
    return next_X


def extract_plans(solverout, n_controlled, n_inter):
    """extract solver output"""
    X_plan = np.zeros((params.n_states * n_controlled, params.N))
    dds_plan = np.zeros((params.n_cinputs * n_controlled, params.N))
    bin_plan = np.zeros((params.n_binputs * n_inter, params.N))
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            X_plan[params.n_states * p_idx:params.n_states * (p_idx + 1), stage_idx] = \
                solverout['x{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1)]
            dds_plan[params.n_cinputs * p_idx:params.n_cinputs * (p_idx + 1), stage_idx] = \
                solverout['u{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1)]
        for i_idx in range(n_inter):
            bin_plan[params.n_binputs * i_idx:params.n_binputs * (i_idx + 1), stage_idx] = \
                solverout['bin{stage:02d}{inter:01d}'.format(stage=stage_idx + 1, inter=i_idx + 1)]

    return X_plan, dds_plan, bin_plan


def check_status(bin_init, X_init, X_curr, players):
    """
    check if all vehicles have passed the intersections and have velocities not less than the initial velocity
    binary variable order: (p1,p2), (p1,p3), (p2.p3)
    """
    task_status = {}
    deadlock = True
    for player in players:
        vx_idx = player_idx[player]*params.n_states + x_idx.dS - params.n_cinputs
        if X_curr[vx_idx]>0.001:
            deadlock = False
        if X_curr[vx_idx] < X_init[vx_idx]:
            task_status[player] = False
        else:
            task_status[player] = True
    if deadlock: 
        for player in players:
            task_status[player] = True
        return task_status
    for i_idx, player_pair in enumerate(combinations(players, 2)):
        player1 = player_pair[0]
        player2 = player_pair[1]
        sigma1_p_idx = i_idx*params.n_binputs + ub_idx.Sigma1_p
        sigma2_p_idx = i_idx*params.n_binputs + ub_idx.Sigma2_p
        if bin_init[sigma1_p_idx] != 1:
            task_status[player1] = False
        if bin_init[sigma2_p_idx] != 1:
            task_status[player2] = False
    return task_status


def update_evaluation(task_status, result, sim_time_idx, dds_plans):
    """record time and total energy consumption of each vehicle when it finishes the task"""
    for player in task_status.keys():
        if task_status[player] and player not in result.keys():
            time = sim_time_idx * params.dt
            dds_idx = player_idx[player]*params.n_cinputs + uc_idx.ddS
            energy = np.linalg.norm(dds_plans[dds_idx, 0, :])
            result[player] = [time, energy]
    return result
