import os

import matplotlib
import matplotlib.pyplot as plt
from geometry import translation_angle_from_SE2
from decimal import Decimal as D

from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons_dev.utils import get_project_root_dir

from trajectory_games.trajectory_generator_dev import TransitionGenerator
from trajectory_games.structures import TrajectoryParams
from trajectory_games.visualization_dev import TrajectoryGenerationVisualization









if __name__ == "__main__":
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    #dgscenario = DgScenario(scenario)
    matplotlib.use("TkAgg")
    #viz = TrajectoryGenerationVisualization(scenario=scenario)
    #viz.plot(show_plot=True, draw_labels=True)
    lanelet_network = scenario.lanelet_network

    points_from_first = 0
    points_from_last = 0

    lane_11 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49564))
    lane_12 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49586))
    lane_13 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49568))

    dglane1_ctrl_points = lane_11.control_points[-points_from_last:-1] \
                          + lane_12.control_points \
                          + lane_13.control_points[1:points_from_first]

    dglanelet = DgLanelet(dglane1_ctrl_points)

    x_1_translation_angles = [translation_angle_from_SE2(dglanelet.center_point(beta)) for
                              beta in range(len(dglanelet.control_points))]

    ref_lane_goals = [RefLaneGoal(ref_lane=dglanelet, goal_progress=0.9)]
    init_point = x_1_translation_angles[1]
    initial_state = VehicleState(x=init_point[0][0]+1, y=init_point[0][1], vx=0, theta=init_point[1], delta=0)
    u_acc = frozenset([-1.0, 0.0, 1.0, 2.0, 3.0])
    u_dst = frozenset([_ * 0.2 for _ in u_acc])
    params = TrajectoryParams(
        solve=False,
        s_final=-1.,
        max_gen=10,
        dt=D("2"),
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=15.0,
        v_min=0.0,
        st_max=0.5,
        dst_max=1.0,
        dt_samp=D("0.1"),
        dst_scale=False,
        n_factor=0.8,
        vg=VehicleGeometry.default_car(),
    )
    generator = TransitionGenerator(params=params, ref_lane_goals=ref_lane_goals)
    trajectories = generator.get_actions(state=initial_state)

    viz = TrajectoryGenerationVisualization(scenario=scenario, trajectories=trajectories)


    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "out/trajectory_generation_test.png")
    viz.plot(show_plot=True, action_color="red", filename=filename)


    # save fig
