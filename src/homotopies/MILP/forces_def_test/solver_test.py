import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons import PlayerName
from homotopies.MILP.utils.prediction import predict, traj2lane, traj_from_commonroad
from dg_commons.sim.scenarios import load_commonroad_scenario
from homotopies.MILP.utils.intersects import find_intersects
from homotopies.MILP.forces_def.sim import get_init_bin, update_sim, extract_plans, check_status, update_evaluation
from homotopies.MILP.forces_def.generate_solver import generate_solver
from homotopies.MILP.forces_def.inequalities import get_ineq
from homotopies.MILP.forces_def.report import *
from homotopies.MILP.forces_def.parameters import params, inter_idx
from homotopies.MILP.forces_def.forces_utils import ForcesException
from reprep import Report

# scenario settings
n_player = 3
p3_from_commonroad = True
player1 = PlayerName('p1')
player2 = PlayerName('p2')
player3 = PlayerName('p3')

state1 = VehicleState(x=-14, y=-70, theta=np.deg2rad(64), vx=3.5, delta=0)
state2 = VehicleState(x=15, y=-52, theta=np.deg2rad(153), vx=4, delta=0)
state3 = VehicleState(x=-8, y=10, theta=0, vx=4, delta=0)

if n_player == 2 or (n_player == 3 and p3_from_commonroad):
    obs = {player1: state1, player2: state2}
else:
    obs = {player1: state1, player2: state2, player3: state3}

scenario_name = "USA_Lanker-2_10_T-1"
scenario_dir = "/home/ysli/Desktop/SP/driving-games/scenarios/"
scenario, _ = load_commonroad_scenario(scenario_name, scenario_dir)
traj_p3 = traj_from_commonroad(scenario_name, scenario_dir, 2653, offset=(0, 0))
vx_p3 = 3

trajs = predict(obs)

if n_player == 3 and p3_from_commonroad:
    trajs[player3] = traj_p3

intersects = find_intersects(trajs)

# matplotlib.use('TkAgg')
# fig, ax = plt.subplots(1, 1)
# colors = {player1: 'blue', player2: 'green', player3: 'black'}
# visualize_trajs_all(trajs, intersects,ax,colors)
# plt.show()

# generate solver
# n_player = len(obs.keys())
n_controlled = n_player
n_inter = int(n_player * (n_player - 1) / 2)

use_bin_init = True
generate_solver(n_player, n_controlled, n_inter, use_bin_init)

import test_py

problem = test_py.test_params

# simulation
sim_time = 60
if n_player == 2:
    x0 = np.array([0, state1.vx, 0, state2.vx])
elif n_player == 3 and not p3_from_commonroad:
    x0 = np.array([0, state1.vx, 0, state2.vx, 0, state3.vx])
else:
    x0 = np.array([0, state1.vx, 0, state2.vx, 0, vx_p3])

X = np.zeros((params.n_states * n_controlled, sim_time + 1))
X_plans = np.zeros((params.n_states * n_controlled, params.N + 1, sim_time))
dds_plans = np.zeros((params.n_cinputs * n_controlled, params.N, sim_time))
bin_plans = np.zeros((params.n_binputs * n_inter, params.N, sim_time))
solvetime = np.zeros(sim_time)
result = {}
X[:, 0] = x0

h = {player1: {player2: 1, player3: 1}, player2: {player3: 1}}

box_buffer = 1.5
A, b = get_ineq(n_controlled, n_inter, trajs, intersects, h, box_buffer=box_buffer)

# simulation
for k in range(sim_time):
    X_plans[:, 0, k] = X[:, k]
    bin_init = get_init_bin(X[:, k], trajs, intersects, n_inter, box_buffer)
    # check if all intersections are passed
    task_status = check_status(bin_init, X[:, 0], X[:, k], trajs.keys())
    result = update_evaluation(task_status, result, k, dds_plans)
    if all(status for status in task_status.values()):  # if all tasks are finished, stop early
        X_plans = X_plans[:, :, :k+1]
        dds_plans = dds_plans[:, :, :k]
        bin_plans = bin_plans[:, :, :k]
        solvetime = solvetime[:k]
        break

    #  provide initial values
    if use_bin_init:
        problem['minus_x0'] = -np.append(bin_init, X[:, k])
    else:
        problem['minus_x0'] = -X[:, k]

    for j in range(0, params.N):
        problem['ineq_A{:02d}'.format(j + 1)] = A
        problem['ineq_b{:02d}'.format(j + 1)] = b
    [solverout, exitflag, info] = test_py.test_solve(problem)

    if (exitflag == 1):
        X_plan, dds_plan, bin_plan = extract_plans(solverout, n_controlled, n_inter)
        X_plans[:, 1:, k] = X_plan
        dds_plans[:, :, k] = dds_plan
        bin_plans[:, :, k] = bin_plan
        solvetime[k] = info.solvetime * 1000
        print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
    else:
        print('exitflag=', exitflag)
        X_plans = X_plans[:, :, :k]
        dds_plans = dds_plans[:, :, :k]
        bin_plans = bin_plans[:, :, :k]
        solvetime = solvetime[:k]
        break
        # raise ForcesException(exitflag)

    X[:, k + 1] = update_sim(X[:, k], dds_plan[:, 0], n_controlled)

# plotting
# matplotlib.use('TkAgg')
# fig, ax = plt.subplots(1, 1)
# visualize_s_plan(X_plans, trajs, intersects, player1, player2, ax)
# plt.show()
colors = {player1: 'blue', player2: 'green', player3: 'black'}
r = Report('test_solver')
r.add_child(get_open_loop_animation(trajs, X_plans, colors, scenario))
r.add_child(generate_report_s_traj(X_plans, trajs, intersects, buffer=box_buffer))
r.add_child(generate_report_input(dds_plans, n_controlled))
r.add_child(generate_report_ds(X_plans, n_controlled))
r.add_child(generate_report_solvetime(solvetime))
r.add_child(generate_report_evaluation(result))
# report_file = os.path.join("src/homotopies/MILP/forces_def_test", f"test_solver")
report_file = "test_solver"
r.to_html(report_file)
