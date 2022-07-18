import os

import matplotlib
import numpy as np
from decimal import Decimal as D

from commonroad.common.solution import VehicleType, vehicle_parameters

from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models import CAR
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.utils import dglane_from_position
from driving_games.utils import get_project_root_dir
from trajectory_games import TrajectoryGenParams

from trajectory_games.trajectory_generator import TrajectoryGenerator
from trajectory_games.visualization import TrajectoryGenerationVisualization


def traj_gen_params_from_cr(cr_vehicle_params, is_ego: bool) -> TrajectoryGenParams:
    vp = VehicleParameters(
        vx_limits=(0.0, cr_vehicle_params.longitudinal.v_max),  # don't allow backwards driving
        acc_limits=(-cr_vehicle_params.longitudinal.a_max, cr_vehicle_params.longitudinal.a_max),
        delta_max=cr_vehicle_params.steering.max,
        ddelta_max=cr_vehicle_params.steering.v_max,
    )

    v_switch = cr_vehicle_params.longitudinal.v_switch

    if is_ego:
        u_acc = frozenset([-1.0, -2.0, -3.0])
        u_dst = frozenset([-0.3, 0.0, 0.3])

    else:
        u_acc = frozenset([-1.0, -2.0, -3.0])
        u_dst = frozenset([-0.3, 0.0, 0.3])

    vg = VehicleGeometry(
        vehicle_type=CAR,
        m=1500.0,
        Iz=1300,
        w_half=cr_vehicle_params.w / 2.0,
        lf=cr_vehicle_params.a,
        lr=cr_vehicle_params.b,
        c_drag=0.3756,
        a_drag=2,
        e=0.5,
        color="royalblue",
    )

    params = TrajectoryGenParams(
        solve=False,
        s_final=-1,
        max_gen=7,
        dt=D("1.0"),
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=vp.vx_limits[1],
        v_min=vp.vx_limits[0],
        st_max=vp.delta_max,
        dst_max=vp.ddelta_max,
        dt_samp=D("0.1"),
        dst_scale=False,
        n_factor=1.0,
        vg=vg,
        acc_max=vp.acc_limits[1],
        v_switch=v_switch,
    )

    return params


def test_trajectory_generation():
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "src/static_scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    # matplotlib.use("TkAgg")

    p = np.array([42.0, 0.0])
    dglane = dglane_from_position(p, scenario.lanelet_network, succ_lane_selection=0)

    ref_lane_goals = [RefLaneGoal(ref_lane=dglane, goal_progress=1000)]

    initial_state = VehicleState(x=p[0] + 10, y=p[1], vx=7, psi=-0.02, delta=-0.02)

    # issues when u_acc <= 0.0
    u_acc = frozenset([-1.0, -5.0])
    u_dst = frozenset([-0.5, 0.0, 0.5])
    # u_dst = frozenset([_ * 0.2 for _ in u_acc])

    # needed for feasibility test
    vg = VehicleGeometry.default_car()
    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_params = vehicle_parameters[vehicle_type]

    params = traj_gen_params_from_cr(vehicle_params, is_ego=False)
    params.u_acc = u_acc
    params.u_dst = u_dst
    params.solve = False
    params.max_gen = 5
    params.dst_scale = False
    params.n_factor = 1.0

    generator = TrajectoryGenerator(params=params, ref_lane_goals=ref_lane_goals)

    trajectories = generator.get_actions(state=initial_state, return_graphs=False)
    # for traj in trajectories:
    #     print(check_feasibility(traj))

    viz = TrajectoryGenerationVisualization(
        scenario=scenario,
        trajectories=trajectories,
        ref_lane_goal=ref_lane_goals[0],
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "out/trajectory_generation_test_0.8.png")
    viz.plot(show_plot=False, draw_labels=True, action_color="red", filename=filename)
    return 0


if __name__ == "__main__":
    test_trajectory_generation()
