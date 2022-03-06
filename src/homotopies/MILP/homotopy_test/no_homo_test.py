import numpy as np
from reprep import Report
from dg_commons.sim.models.vehicle import VehicleState

from homotopies.MILP.forces_def.generate_solver import generate_solver
from homotopies.MILP.forces_def.parameters import params, x_idx
from homotopies.MILP.forces_def.report import generate_report_solver
from homotopies.MILP.forces_def.sim import sim
from homotopies.MILP.homotopy.report import generate_homotopy_report
from homotopies.MILP.scenario import get_simple_scenario, get_commonroad_scenario
from homotopies.MILP.utils.prediction import predict
from homotopies.MILP.utils.intersects import find_intersects
from homotopies.MILP.homotopy.homotopy import *

# scenario settings
player1 = PlayerName('p1')
player2 = PlayerName('p2')
player3 = PlayerName('p3')
n_player = 3
# trajs, intersects, x0, scenario = get_simple_scenario(n_player)
trajs, intersects, x0, scenario = get_commonroad_scenario()
vx_ref = np.zeros(n_player)
for p_idx in range(n_player):
    vx_ref[p_idx] = x0[p_idx*params.n_states+x_idx.dS-params.n_cinputs]

# generate solver
n_controlled = n_player
n_inter = int(n_player * (n_player - 1) / 2)
use_bin_init = True
use_homo = False
solver_name = generate_solver(n_player, n_controlled, n_inter, use_bin_init, use_homo)
# solver_name = 'test'
module_name = solver_name + '_py'

r = Report('no_homotopy_test')
h = None
X_plans, dds_plans, bin_plans, solvetime, performance = sim(module_name, n_controlled, n_inter, trajs, intersects, x0, h, use_bin_init, use_homo)
# generate report
colors = {player1: 'blue', player2: 'green', player3: 'black'}
r.add_child(generate_report_solver(n_controlled, trajs, intersects, X_plans, dds_plans, solvetime, performance, h, colors,
                           scenario))
report_file = "no_homotopy_test"
r.to_html(report_file)
