import numpy as np
from reprep import Report
from dg_commons.sim.models.vehicle import VehicleState

from homotopies.MIQP.forces_def.generate_solver import generate_solver
from homotopies.MIQP.forces_def.parameters import params, x_idx
from homotopies.MIQP.forces_def.report import generate_report_solver
from homotopies.MIQP.forces_def.sim import sim
from homotopies.MIQP.homotopy.report import generate_homotopy_report
from homotopies.MIQP.scenario import get_simple_scenario, get_commonroad_scenario
from homotopies.MIQP.utils.prediction import predict
from homotopies.MIQP.utils.intersects import find_intersects
from homotopies.MIQP.homotopy.homotopy import *

"""scenario settings"""
player1 = PlayerName("p1")
player2 = PlayerName("p2")
player3 = PlayerName("p3")
n_player = 3
# trajs, intersects, x0, scenario = get_simple_scenario(n_player)
trajs, intersects, x0, scenario = get_commonroad_scenario()
vx_ref = np.zeros(n_player)
for p_idx in range(n_player):
    vx_ref[p_idx] = x0[p_idx * params.n_states + x_idx.dS - params.n_cinputs]

"""generate solver(under one homotopy class)"""
n_controlled = n_player
n_inter = int(n_player * (n_player - 1) / 2)
use_bin_init = True
use_homo = True
solver_name = generate_solver(n_player, n_controlled, n_inter, use_bin_init, use_homo)
module_name = solver_name + "_py"

"""evaluate and rank all homotopy classes, apply the solver to all homotopy classes"""
homotopies_sorted = evaluate_homotopy(intersects, trajs.keys(), vx_ref)
r = Report("homotopy_test")
r.add_child(generate_homotopy_report(homotopies_sorted))
for homotopy in homotopies_sorted:
    X_plans, dds_plans, bin_plans, solvetime, performance = sim(
        module_name, n_controlled, n_inter, trajs, intersects, x0, homotopy, use_bin_init
    )
    # generate report
    colors = {player1: "blue", player2: "green", player3: "black"}
    report_name = ""
    for b in homotopy.h:
        report_name += str(b)
    r.add_child(
        generate_report_solver(
            report_name, n_controlled, trajs, intersects, X_plans, dds_plans, solvetime, performance, colors, scenario
        )
    )
    report_file = "homotopy_test"
    r.to_html(report_file)
