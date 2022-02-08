import os
from typing import Mapping
from decimal import Decimal as D
from dataclasses import replace

import numpy as np

from dg_commons import PlayerName
from dg_commons.planning import JointTrajectories, Trajectory, RefLaneGoal
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons_dev.utils import get_project_root_dir
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.maps import DgLanelet, LaneCtrPoint, SE2Transform
from driving_games.metrics_structures import MetricEvaluationContext
from trajectory_games.visualization_dev import EvaluationContextVisualization
from trajectory_games.metrics import EpisodeTime, DeviationLateral, DeviationHeading
from math import pi

P1 = PlayerName("p1")
P2 = PlayerName("p2")
P3 = PlayerName("p3")


def get_default_evaluation_context() -> MetricEvaluationContext:
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    dgscenario = DgScenario(scenario)

    # todo create some fake joint trajectories and goals

    joint_trajectories: JointTrajectories = {}
    # reference_trajectories: Mapping[PlayerName, Trajectory] = {} #question: is this needed? Ask Ale is ok to change interface
    goals: Mapping[PlayerName, RefLaneGoal] = {}

    # todo: add goals and reference trajectories here
    # todo add references later (?)

    """
    class RefLaneGoal(PlanningGoal):
        ref_lane: DgLanelet
        goal_progress: float
    """

    """
    goals = {P1: RefLaneGoal(ref_lane=, goal_progress=0.5),
             P2: RefLaneGoal(ref_lane=, goal_progress=0.5),
             P3: RefLaneGoal(ref_lane=, goal_progress=0.5)}
    """
    lanelet_network = scenario.lanelet_network

    lanelets_l1 = [lanelet_network.find_lanelet_by_id(49570).center_vertices,
                   lanelet_network.find_lanelet_by_id(49598).center_vertices,
                   lanelet_network.find_lanelet_by_id(49576).center_vertices]

    lanelets_l2 = [lanelet_network.find_lanelet_by_id(49574).center_vertices,
                   lanelet_network.find_lanelet_by_id(49600).center_vertices,
                   lanelet_network.find_lanelet_by_id(49566).center_vertices]

    lanelets_l3 = [lanelet_network.find_lanelet_by_id(49564).center_vertices,
                   lanelet_network.find_lanelet_by_id(49586).center_vertices,
                   lanelet_network.find_lanelet_by_id(49568).center_vertices]

    points_kept_start = 4
    points_kept_end = 4

    lane_center_points_1 = np.append(np.append(lanelets_l1[0][-points_kept_end:],
                                               lanelets_l1[1], axis=0),
                                     lanelets_l1[2][:points_kept_start], axis=0)

    lane_center_points_2 = np.append(np.append(lanelets_l2[0][-points_kept_end:],
                                               lanelets_l2[1], axis=0),
                                     lanelets_l2[2][:points_kept_start], axis=0)

    lane_center_points_3 = np.append(np.append(lanelets_l3[0][-points_kept_end:],
                                               lanelets_l3[1], axis=0),
                                     lanelets_l3[2][:points_kept_start], axis=0)

    radius = 2.0
    lane_ctr_points_1 = [LaneCtrPoint(SE2Transform(p=point, theta=0), r=radius) for point in lane_center_points_1]
    lane_ctr_points_2 = [LaneCtrPoint(SE2Transform(p=point, theta=0), r=radius) for point in lane_center_points_2]
    lane_ctr_points_3 = [LaneCtrPoint(SE2Transform(p=point, theta=0), r=radius) for point in lane_center_points_3]

    goals = {P1: RefLaneGoal(ref_lane=DgLanelet(lane_ctr_points_1), goal_progress=0.8),  # todo: feed radius?
             P2: RefLaneGoal(ref_lane=DgLanelet(lane_ctr_points_2), goal_progress=0.8),
             P3: RefLaneGoal(ref_lane=DgLanelet(lane_ctr_points_3), goal_progress=0.8)}

    x_offset_1 = 2.0
    y_offset_1 = 1.5
    theta_offset_2 = pi / 3

    x_1 = [VehicleState(x=point[0] + x_offset_1, y=point[1], theta=0, vx=2.0, delta=0.0) for point in
           lane_center_points_1]
    x_2 = [VehicleState(x=point[0], y=point[1] + y_offset_1, theta=pi / 2, vx=1.0, delta=0.0) for point in
           lane_center_points_2]
    x_3 = [VehicleState(x=point[0], y=point[1], theta=-pi + theta_offset_2, vx=3.0, delta=0.0) for point in
           lane_center_points_3]

    t_max = 10.0
    joint_trajectories = {
        P1: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_1))), values=x_1),
        P2: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_2))), values=x_2),
        P3: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_3))), values=x_3)
    }

    return MetricEvaluationContext(dgscenario=dgscenario, trajectories=joint_trajectories, goals=goals)


def visualize_evaluation_context(context: MetricEvaluationContext):
    colors = {P1: 'red', P2: 'blue', P3: 'pink'}
    viz = EvaluationContextVisualization(evaluation_context=context)
    viz.plot(show_plot=True, draw_labels=False, action_colors=colors)
    return


def test_metrics_1():
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context)
    episode_time = EpisodeTime()
    episode_times = episode_time.evaluate(context=evaluation_context)
    print("Episode Times: " + str(episode_times))

    deviation_lateral = DeviationLateral()
    deviations_lat = deviation_lateral.evaluate(context=evaluation_context)
    print("Deviation Headings: " + str(deviations_lat))

    # todo


def test_metrics_2():
    evaluation_context = get_default_evaluation_context()
    # todo


if __name__ == "__main__":
    test_metrics_1()
