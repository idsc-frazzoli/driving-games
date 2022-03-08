from dg_commons import PlayerName
from homotopies.MILP.forces_def.generate_solver import generate_solver
from homotopies.MILP.forces_def.report import generate_report_solver
from homotopies.MILP.forces_def.sim import sim
from homotopies.MILP.scenario import get_simple_scenario, get_commonroad_scenario
from homotopies.MILP.homotopy.homotopy import Homotopy

"""scenario settings"""
player1 = PlayerName('p1')
player2 = PlayerName('p2')
player3 = PlayerName('p3')
n_player = 3
# trajs, intersects, x0, scenario = get_simple_scenario(n_player)
trajs, intersects, x0, scenario = get_commonroad_scenario()
"""generate solver"""
n_controlled = n_player
n_inter = int(n_player * (n_player - 1) / 2)
use_bin_init = True

solver_name = generate_solver(n_player, n_controlled, n_inter, use_bin_init)
# solver_name = 'test'
module_name = solver_name + '_py'

"""start simulation"""
h = [1, 0, 0]#{player1: {player2: 1, player3: 0}, player2: {player3: 0}}
homotopy = Homotopy(intersects=intersects, players=trajs.keys(), vx_ref=None, h=h)
X_plans, dds_plans, bin_plans, solvetime, performance = sim(module_name, n_controlled, n_inter, trajs, intersects, x0, homotopy, use_bin_init)

""" generate report"""
colors = {player1: 'blue', player2: 'green', player3: 'black'}
r = generate_report_solver(n_controlled, trajs, intersects, X_plans, dds_plans, solvetime, performance, homotopy, colors, scenario)
report_file = "best_homotopy"
r.to_html(report_file)
