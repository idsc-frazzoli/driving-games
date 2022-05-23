import os

import matplotlib
import numpy as np
from decimal import Decimal as D

from commonroad.common.solution import VehicleType, vehicle_parameters

from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons_dev.utils import get_project_root_dir

from trajectory_games.trajectory_generator import TrajectoryGenerator
from trajectory_games.structures import TrajectoryGenParams
from trajectory_games.visualization import TrajectoryGenerationVisualization


def test_trajectory_generation():
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    matplotlib.use("TkAgg")

    p = np.array([42.0, 0.0])
    dglane = dglane_from_position(p, scenario.lanelet_network, succ_lane_selection=0)

    ref_lane_goals = [RefLaneGoal(ref_lane=dglane, goal_progress=1000)]

    initial_state = VehicleState(x=p[0]+10, y=p[1], vx=7, theta=-0.02, delta=-0.02)

    # issues when u_acc <= 0.0
    u_acc = frozenset([-1.0, -5.0])
    u_dst = frozenset([-0.5, 0.0, 0.5])
    # u_dst = frozenset([_ * 0.2 for _ in u_acc])

    # needed for feasibility test
    vg = VehicleGeometry.default_car()
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_params = vehicle_parameters[vehicle_type]

    from commonroad_challenge_tests.trajectory_feasibility_test import traj_gen_params_from_cr
    params = traj_gen_params_from_cr(vehicle_params, is_ego=False)
    params.u_acc = u_acc
    params.u_dst = u_dst
    params.solve = False
    params.max_gen = 5
    params.dst_scale = False
    params.n_factor = 1.0

    # params = TrajectoryGenParams(
    #     solve=False,
    #     s_final=-1,  # todo: add this to generator
    #     max_gen=5,
    #     dt=D("1.0"),  # keep at max 1 sec, increase k_maxgen in trajectrory_generator for having more generations
    #     u_acc=u_acc,
    #     u_dst=u_dst,
    #     v_max=15.0,
    #     v_min=0.0,
    #     st_max=0.5,
    #     dst_max=1.0,
    #     dt_samp=D("0.2"),
    #     dst_scale=False,
    #     n_factor=1.0,
    #     v_switch=4.0,
    #     acc_max=11.5,
    #     vg=VehicleGeometry.default_car(),
    # )


    generator = TrajectoryGenerator(params=params, ref_lane_goals=ref_lane_goals)

    trajectories = generator.get_actions(state=initial_state, return_graphs=False)
    from commonroad_challenge_tests.trajectory_feasibility_test import check_feasibility
    # for traj in trajectories:
    #     print(check_feasibility(traj))

    viz = TrajectoryGenerationVisualization(scenario=scenario, trajectories=trajectories, ref_lane_goal=ref_lane_goals[0])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "out/trajectory_generation_test_0.8.png")
    viz.plot(show_plot=True, draw_labels=True, action_color="red", filename=filename)
    return 0


if __name__ == "__main__":
    test_trajectory_generation()
