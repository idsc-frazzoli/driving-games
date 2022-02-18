import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons import PlayerName
from homotopies.MILP.utils.prediction import predict, traj2lane
from homotopies.MILP.utils.intersects import find_intersects
from homotopies.MILP.forces_def.equalities import get_init_bin
from homotopies.MILP.forces_def.generate_solver import generate_solver, extract_plans
from homotopies.MILP.forces_def.inequalities import get_ineq
from homotopies.MILP.forces_def.visualization import *
from homotopies.MILP.forces_def.parameters import params
from homotopies.MILP.forces_def.forces_utils import ForcesException
from reprep import Report


def update_sim(X, ddS, n_controlled):
    next_X = np.zeros_like(X)
    for p_idx in range(n_controlled):
        x_idx = list(range(p_idx*params.n_states, (p_idx+1)*params.n_states))
        dds_idx = list(range(p_idx * params.n_cinputs, (p_idx + 1) * params.n_cinputs))
        curr = np.hstack((ddS[dds_idx], X[x_idx])).reshape([-1, 1])
        next_X[x_idx] = np.dot(params.C, curr).reshape([2, ])
    return next_X

# scenario settings
state1 = VehicleState(x=15, y=0, theta=np.pi / 2, vx=3.5, delta=0.03)
state2 = VehicleState(x=-5, y=0, theta=np.pi / 4, vx=5, delta=-0.03)
state3 = VehicleState(x=-8, y=15, theta=0, vx=4, delta=0)

player1 = PlayerName('p1')
player2 = PlayerName('p2')
player3 = None
# player3 = PlayerName('p3')
if player3 is None:
    obs = {player1: state1, player2: state2}
else:
    obs = {player1: state1, player2: state2, player3: state3}


trajs = predict(obs)
intersects = find_intersects(trajs)

# generate solver
n_player = len(obs.keys())
n_controlled = n_player
n_inter = int(n_player*(n_player-1)/2)

use_bin_init = True
generate_solver(n_player, n_controlled, n_inter, use_bin_init)

import test_py
problem = test_py.test_params

# simulation
sim_time = 30
if player3 is None:
    x0 = np.array([0, state1.vx, 0, state2.vx])
else:
    x0 = np.array([0, state1.vx, 0, state2.vx, 0, state3.vx])

X = np.zeros((params.n_states*n_controlled, sim_time + 1))
X_plans = np.zeros((params.n_states*n_controlled, params.N+1, sim_time))
dds_plans = np.zeros((params.n_cinputs*n_controlled, params.N, sim_time))
bin_plans = np.zeros((params.n_binputs*n_inter, params.N, sim_time))
X[:, 0] = x0

h = {player1: {player2: 1, player3: 1}, player2: {player3: 1}}
box_buffer = 1.5
A, b = get_ineq(n_controlled, n_inter, trajs, intersects, h, box_buffer=box_buffer)  # todo: handle cases when there's no intersect
solvetime = np.zeros(sim_time)

# simulation
for k in range(sim_time):
    init_bin = get_init_bin(X[:, k], trajs, intersects, n_inter, box_buffer)
    if use_bin_init:
        problem['minus_x0'] = -np.append(init_bin, X[:, k])
    else:
        problem['minus_x0'] = -X[:, k]
    for j in range(0, params.N):
        problem['ineq_A{:02d}'.format(j + 1)] = A
        problem['ineq_b{:02d}'.format(j + 1)] = b
    [solverout, exitflag, info] = test_py.test_solve(problem)
    if (exitflag == 1):
        X_plan, dds_plan, bin_plan = extract_plans(solverout, n_controlled, n_inter)
        X_plans[:, 0, k] = X[:, k]
        X_plans[:, 1:, k] = X_plan
        dds_plans[:, :, k] = dds_plan
        bin_plans[:, :, k] = bin_plan
        solvetime[k] = info.solvetime * 1000
        print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
    else:
        print('exitflag=', exitflag)
        raise ForcesException(exitflag)
        # raise RuntimeError('Some problem in solver')

    X[:, k + 1] = update_sim(X[:, k], dds_plan[:, 0], n_controlled)

# plotting
# matplotlib.use('TkAgg')
# fig, ax = plt.subplots(1, 1)
# visualize_s_plan(X_plans, trajs, intersects, player1, player2, ax)
# plt.show()
colors = {player1: 'blue', player2: 'green', player3: 'black'}
r = Report('test_solver')
r.add_child(get_open_loop_animation(trajs, X_plans, colors))
r.add_child(generate_report_s_plan(X_plans, trajs, intersects, player1, player2, buffer=box_buffer))
r.add_child(generate_report_input(dds_plans, n_controlled, sim_time))
r.add_child(generate_report_ds(X_plans, n_controlled, sim_time))
r.add_child(generate_report_solvetime(solvetime, sim_time))
r.to_html('test_report')
