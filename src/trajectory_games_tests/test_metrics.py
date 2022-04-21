import os
from dataclasses import dataclass, field
from math import pi
from typing import Mapping, Optional

import matplotlib
import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficLight, TrafficLightCycleElement, TrafficLightState
from geometry.poses import translation_angle_from_SE2

from crash import logger
from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.planning import JointTrajectories, Trajectory, RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons_dev.utils import get_project_root_dir
from driving_games.metrics_structures import MetricEvaluationContext
from trajectory_games.metrics import *
from trajectory_games.visualization import EvaluationContextVisualization
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

P1 = PlayerName("p1")
P2 = PlayerName("p2")
P3 = PlayerName("p3")

size_p1_trajectory: int = 0
size_p2_trajectory: int = 0
size_p3_trajectory: int = 0


@dataclass
class _PlayerOffsets:
    # size of offset vectors
    size: int = 0

    # default values (i.e. no offset)
    x_default_value: float = 0.0
    y_default_value: float = 0.0
    theta_default_value: float = 0.0
    v_default_value: float = 0.0
    acc_default_value: float = 0.0
    delta_default_value: float = 0.0
    delta_rate_default_value: float = 0.0

    x_offset: list = field(default_factory=list)
    y_offset: list = field(default_factory=list)
    v_offset: list = field(default_factory=list)
    theta_offset: list = field(default_factory=list)
    delta_offset: list = field(default_factory=list)

    def __post_init__(self):
        # generate lists of default offsets if they are not specified
        if not self.x_offset:
            self.x_offset = [self.x_default_value for _ in range(self.size)]
        if not self.y_offset:
            self.y_offset = [self.y_default_value for _ in range(self.size)]
        if not self.v_offset:
            self.v_offset = [self.v_default_value + i * self.acc_default_value for i in range(self.size)]
        if not self.theta_offset:
            self.theta_offset = [self.theta_default_value for _ in range(self.size)]
        if not self.delta_offset:
            self.delta_offset = [self.delta_default_value + i * self.delta_rate_default_value for i in range(self.size)]


JointPlayerOffsets = Mapping[PlayerName, _PlayerOffsets]


def add_traffic_light_custom(scenario: Scenario) -> Scenario:
    green: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.RED, duration=10)
    yellow: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.YELLOW, duration=5)
    red: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.GREEN, duration=10)
    cycle = [green, yellow, red]
    position = np.array([73.0, -8.0])
    traffic_light: TrafficLight = TrafficLight(traffic_light_id=0, cycle=cycle, position=position)
    scenario.add_objects(traffic_light, lanelet_ids={49570})
    return scenario

def get_default_evaluation_context(player_offsets: Optional[JointPlayerOffsets] = None) -> MetricEvaluationContext:
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    dgscenario = DgScenario(scenario)

    # defines reference lanelets for players
    lanelet_network = scenario.lanelet_network

    points_from_first = 4
    points_from_last = 4

    lane_11 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane_12 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49598))
    lane_13 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49576))

    lane_21 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49574))
    lane_22 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49600))
    lane_23 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49566))

    lane_31 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49564))
    lane_32 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49586))
    lane_33 = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49568))

    dglane1_ctrl_points = (
            lane_11.control_points[-points_from_last:-1]
            + lane_12.control_points
            + lane_13.control_points[1:points_from_first]
    )

    dglane2_ctrl_points = (
            lane_21.control_points[-points_from_last:-1]
            + lane_22.control_points
            + lane_23.control_points[1:points_from_first]
    )

    dglane3_ctrl_points = (
            lane_31.control_points[-points_from_last:-1]
            + lane_32.control_points
            + lane_33.control_points[1:points_from_first]
    )

    dglanelet_1 = DgLanelet(dglane1_ctrl_points)
    dglanelet_2 = DgLanelet(dglane2_ctrl_points)
    dglanelet_3 = DgLanelet(dglane3_ctrl_points)

    goals = {
        P1: [RefLaneGoal(ref_lane=dglanelet_1, goal_progress=0.8)],
        P2: [RefLaneGoal(ref_lane=dglanelet_2, goal_progress=0.8)],
        P3: [RefLaneGoal(ref_lane=dglanelet_3, goal_progress=0.8)],
    }

    # Define trajectories for players
    x_1_translation_angles = [
        translation_angle_from_SE2(dglanelet_1.center_point(beta)) for beta in range(len(dglanelet_1.control_points))
    ]
    x_2_translation_angles = [
        translation_angle_from_SE2(dglanelet_2.center_point(beta)) for beta in range(len(dglanelet_2.control_points))
    ]
    x_3_translation_angles = [
        translation_angle_from_SE2(dglanelet_3.center_point(beta)) for beta in range(len(dglanelet_3.control_points))
    ]

    global size_p1_trajectory, size_p2_trajectory, size_p3_trajectory

    size_p1_trajectory = len(x_1_translation_angles)
    size_p2_trajectory = len(x_2_translation_angles)
    size_p3_trajectory = len(x_3_translation_angles)

    if player_offsets is None:
        player_offsets = {
            P1: _PlayerOffsets(size=size_p1_trajectory),
            P2: _PlayerOffsets(size=size_p2_trajectory),
            P3: _PlayerOffsets(size=size_p3_trajectory),
        }

    x_1 = [
        VehicleState(
            x=translation[0] + player_offsets[P1].x_offset[i],
            y=translation[1] + player_offsets[P1].y_offset[i],
            theta=angle + player_offsets[P1].theta_offset[i],
            vx=0.0 + player_offsets[P1].v_offset[i],
            delta=0.0 + player_offsets[P1].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_1_translation_angles)
    ]

    x_2 = [
        VehicleState(
            x=translation[0] + player_offsets[P2].x_offset[i],
            y=translation[1] + player_offsets[P2].y_offset[i],
            theta=angle + player_offsets[P2].theta_offset[i],
            vx=0.0 + player_offsets[P2].v_offset[i],
            delta=0.0 + player_offsets[P2].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_2_translation_angles)
    ]

    x_3 = [
        VehicleState(
            x=translation[0] + player_offsets[P3].x_offset[i],
            y=translation[1] + player_offsets[P3].y_offset[i],
            theta=angle + player_offsets[P3].theta_offset[i],
            vx=0.0 + player_offsets[P3].v_offset[i],
            delta=0.0 + player_offsets[P3].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_3_translation_angles)
    ]

    t_max = 10.0
    joint_trajectories: JointTrajectories = {
        P1: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_1))), values=x_1),
        P2: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_2))), values=x_2),
        P3: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_3))), values=x_3),
    }

    geos = {
        P1: VehicleGeometry.default_car(),
        P2: VehicleGeometry.default_car(),
        P3: VehicleGeometry.default_car(),
    }

    return MetricEvaluationContext(dgscenario=dgscenario, trajectories=joint_trajectories, goals=goals, geos=geos)


def visualize_evaluation_context(context: MetricEvaluationContext, show_plot: bool = True):
    colors = {P1: "red", P2: "blue", P3: "pink"}
    viz = EvaluationContextVisualization(evaluation_context=context)
    viz.plot(show_plot=show_plot, draw_labels=True, action_colors=colors)
    return


def test_times():
    show_plots = True
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    episode_time = EpisodeTime()
    episode_times = episode_time.evaluate(context=evaluation_context)
    logger.info(f"Test 1 executed with results: \n")
    logger.info(f"Episode Times: {episode_times}")
    logger.info(f"Test 1 finished.")
    assert episode_times[P1].value == 10.0
    assert episode_times[P2].value == 10.0
    assert episode_times[P3].value == 10.0


def test_lateral_deviation():
    show_plots = False
    deviation_lateral = DeviationLateral()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    deviations_lat_0 = deviation_lateral.evaluate(context=evaluation_context)
    logger.info(f"Test Lateral Deviation results:")
    logger.info(f"No Offset.")
    logger.info(f"Lateral Deviation: {deviations_lat_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=1.0, y_default_value=0, theta_default_value=0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=1.0, y_default_value=0, theta_default_value=0),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=1.0, y_default_value=1.0, theta_default_value=0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    deviations_lat_1 = deviation_lateral.evaluate(context=evaluation_context_1)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Lateral Deviation: {deviations_lat_1}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=0, y_default_value=1.0, theta_default_value=0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=0, y_default_value=1.0, theta_default_value=0),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=-1.0, y_default_value=-1.0, theta_default_value=0),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    deviations_lat_2 = deviation_lateral.evaluate(context=evaluation_context_2)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Lateral Deviation: {deviations_lat_2}")
    logger.info(f"--------------------------------------------")

    joint_player_offsets_3 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=-1.0, y_default_value=0.0, theta_default_value=0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=0),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=2.0, y_default_value=2.0, theta_default_value=0),
    }

    evaluation_context_3 = get_default_evaluation_context(joint_player_offsets_3)
    visualize_evaluation_context(context=evaluation_context_3, show_plot=show_plots)
    deviations_lat_3 = deviation_lateral.evaluate(context=evaluation_context_3)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_3)
    logger.info(f"Lateral Deviation: {deviations_lat_3}")
    logger.info(f"--------------------------------------------")
    logger.info(f"Test Lateral Deviation finished.")

    # scenario 1 vs scenario 0
    # All lateral deviations should be greater in 1 since in 0 there is no offset
    assert deviations_lat_1[P1].value > deviations_lat_0[P1].value
    assert deviations_lat_1[P2].value > deviations_lat_0[P2].value
    assert deviations_lat_1[P3].value > deviations_lat_0[P3].value

    # scenario 2 vs scenario 0
    # All lateral deviations should be greater in 2 since in 0 there is no offset
    assert deviations_lat_2[P1].value > deviations_lat_0[P1].value
    assert deviations_lat_2[P2].value > deviations_lat_0[P2].value
    assert deviations_lat_2[P3].value > deviations_lat_0[P3].value

    # scenario 3 vs scenario 0
    # Lateral deviations for P1&P3 should be greater in 1 since in 0 there is no offset
    # Lateral deviation for P2 should be equal
    assert deviations_lat_3[P1].value > deviations_lat_0[P1].value
    assert deviations_lat_3[P2].value == deviations_lat_0[P2].value
    assert deviations_lat_3[P3].value > deviations_lat_0[P3].value

    # scenario 1 vs scenario 2
    # P1 has vertical trajectory (x approx constant) -> offset in x should increase metric more
    # P2 has horizontal trajectory (y approx constant) -> offset in y should increase metric more
    # P3 had 90-degree angle curve -> offset (-1,-1) and (1,1) should be similar
    assert deviations_lat_1[P1].value > deviations_lat_2[P1].value
    assert deviations_lat_2[P2].value > deviations_lat_1[P2].value
    assert abs(deviations_lat_1[P3].value - deviations_lat_2[P3].value) < 1.0

    # scenario 1 vs scenario 3
    # P1 has vertical trajectory (x approx constant) -> offset in y should not change the metric too much
    #   high tolerance since moving the curves of P1 horizontally makes them different at bending point
    # P2: Offset only in scenario 1
    # P3 : higher deviations both in x and y should lead to higher metric
    assert abs(deviations_lat_1[P1].value - deviations_lat_3[P1].value) < 2.0
    assert deviations_lat_1[P2].value > deviations_lat_3[P2].value
    assert deviations_lat_3[P3].value > deviations_lat_1[P3].value

    # scenario 2 vs scenario 3
    # P1 has vertical trajectory (x approx constant) -> offset in x should increase metric more (testing negative
    #   offset)
    # P2: Offset only in scenario 2
    # P3: higher deviations both in x and y should lead to higher metric (testing negative offsets)
    assert deviations_lat_3[P1].value > deviations_lat_2[P1].value
    assert deviations_lat_2[P2].value > deviations_lat_3[P2].value
    assert deviations_lat_3[P3].value > deviations_lat_2[P3].value


def test_heading_deviation():
    show_plots = False
    deviation_heading = DeviationHeading()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    deviations_head_0 = deviation_heading.evaluate(context=evaluation_context)
    logger.info(f"Test Heading Deviation results:")
    logger.info(f"No Offset.")
    logger.info(f"Heading Deviations: {deviations_head_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=pi / 6
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=-pi / 6
        ),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=pi / 8
        ),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    deviations_head_1 = deviation_heading.evaluate(context=evaluation_context_1)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Heading Deviations: {deviations_head_1}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=-pi / 6
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory, x_default_value=0.0, y_default_value=2.0, theta_default_value=-pi / 6
        ),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=2.0, y_default_value=2.0, theta_default_value=pi / 8
        ),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    deviations_head_2 = deviation_heading.evaluate(context=evaluation_context_2)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Heading Deviations: {deviations_head_2}")
    logger.info(f"--------------------------------------------")

    joint_player_offsets_3 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory, x_default_value=-1.0, y_default_value=0.0, theta_default_value=pi / 3
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=-pi / 3
        ),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=2.0, y_default_value=2.0, theta_default_value=pi / 4
        ),
    }

    evaluation_context_3 = get_default_evaluation_context(joint_player_offsets_3)
    visualize_evaluation_context(context=evaluation_context_3, show_plot=show_plots)
    deviations_head_3 = deviation_heading.evaluate(context=evaluation_context_3)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_3)
    logger.info(f"Heading Deviations: {deviations_head_3}")
    logger.info(f"--------------------------------------------")

    # scenario 0
    # All heading deviations should be close to zero
    assert abs(deviations_head_0[P1].value - 0.0) < 0.6
    assert abs(deviations_head_0[P2].value - 0.0) < 0.6
    assert abs(deviations_head_0[P3].value - 0.0) < 0.6

    # scenario 1 vs scenario 0
    # All lateral deviations should be greater in 1 since in 0 there is no offset
    assert deviations_head_1[P1].value > deviations_head_0[P1].value
    assert deviations_head_1[P2].value > deviations_head_0[P2].value
    assert deviations_head_1[P3].value > deviations_head_0[P3].value

    # scenario 2 vs scenario 0
    # All heading deviations should be greater in 2 since in 0 there is no offset
    assert deviations_head_2[P1].value > deviations_head_0[P1].value
    assert deviations_head_2[P2].value > deviations_head_0[P2].value
    assert deviations_head_2[P3].value > deviations_head_0[P3].value

    # scenario 3 vs scenario 0
    # All heading deviations should be greater in 2 since in 0 there is no offset
    assert deviations_head_3[P1].value > deviations_head_0[P1].value
    assert deviations_head_3[P2].value > deviations_head_0[P2].value
    assert deviations_head_3[P3].value > deviations_head_0[P3].value

    # scenario 1 vs scenario 2
    # P1: inverting sign of theta should yield a similar heading deviation
    # P2: translating trajectory but not changing heading should yield similar heading deviation
    # P3: translating trajectory but not changing heading should yield similar heading deviation
    assert abs(deviations_head_1[P1].value - deviations_head_2[P1].value) < 0.6
    assert abs(deviations_head_1[P2].value - deviations_head_2[P2].value) < 0.6
    assert abs(deviations_head_1[P3].value - deviations_head_2[P3].value) < 0.6

    # scenario 1 vs scenario 3
    # greater heading angles should lead to greater heading deviation metric
    assert deviations_head_3[P1].value > deviations_head_1[P1].value
    assert deviations_head_3[P2].value > deviations_head_1[P2].value
    assert deviations_head_3[P3].value > deviations_head_1[P3].value

    logger.info(f"Test Heading Deviations finished.")


def test_drivable_area_violation():
    show_plots = False
    area_violation = DrivableAreaViolation()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    area_violation_0 = area_violation.evaluate(context=evaluation_context)
    logger.info(f"Test Drivable area violation results:")
    logger.info(f"No Offset.")
    logger.info(f"Drivable area violation: {area_violation_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=0.0, y_default_value=2.0, theta_default_value=0.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=2.0, y_default_value=0.0, theta_default_value=0.0),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=-3.0, y_default_value=-3.0, theta_default_value=0.0
        ),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    area_violation_1 = area_violation.evaluate(context=evaluation_context_1)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Drivable area violation: {area_violation_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    # All heading deviations should be close to zero
    assert abs(area_violation_0[P1].value - 0.0) < 0.1
    assert abs(area_violation_0[P2].value - 0.0) < 0.1
    assert abs(area_violation_0[P3].value - 0.0) < 0.1

    # scenario 1 vs scenario 0
    # P1: offset in y should not lead to area violation
    # P2: offset in x should not lead to area violation
    # P3: area violation in scenario 1 should be greater than in scenario 0
    assert area_violation_0[P1].value == 0.0
    assert area_violation_0[P2].value == 0.0
    assert area_violation_1[P3].value > area_violation_0[P3].value

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=3.0, y_default_value=0.0, theta_default_value=0.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=0.0, y_default_value=3.0, theta_default_value=0.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=2.0, y_default_value=2.0, theta_default_value=0.0),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    area_violation_2 = area_violation.evaluate(context=evaluation_context_2)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Drivable area violation: {area_violation_2}")
    logger.info(f"--------------------------------------------\n")

    # Scenario 2 vs scenario 0
    # All: area violation in scenario 2 should be greater than in scenario 0
    assert area_violation_2[P1].value > area_violation_0[P1].value
    assert area_violation_2[P2].value > area_violation_0[P2].value
    assert area_violation_2[P3].value > area_violation_0[P3].value

    joint_player_offsets_3 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=pi / 3
        ),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=0.0, y_default_value=-6.0, theta_default_value=0.0),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=-6.0, y_default_value=-6.0, theta_default_value=0.0
        ),
    }

    evaluation_context_3 = get_default_evaluation_context(joint_player_offsets_3)
    visualize_evaluation_context(context=evaluation_context_3, show_plot=show_plots)
    area_violation_3 = area_violation.evaluate(context=evaluation_context_3)
    logger.info(f"Offset:")
    logger.info(joint_player_offsets_3)
    logger.info(f"Drivable area violation: {area_violation_3}")
    logger.info(f"--------------------------------------------\n")

    # scenario 3
    # P1: offset in theta should not impact drivable area violation
    # P2 & P3: offset in negative direction but greater magnitude should lead to greater area violation than
    # with smaller magnitude of offset (scenario 2)
    assert area_violation_3[P1].value == area_violation_0[P1].value
    assert area_violation_3[P2].value > area_violation_2[P2].value
    assert area_violation_3[P3].value > area_violation_2[P3].value

    logger.info(f"Test Drivable Area Violation finished.")


def test_progress_along_reference():
    show_plots = False
    progress = ProgressAlongReference()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    progress_0 = progress.evaluate(context=evaluation_context)
    logger.info(f"Test progress along reference results:")
    logger.info(f"No Offset.")
    logger.info(f"Progress along reference: {progress_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=0.0, y_default_value=2.0, theta_default_value=0.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=2.0, y_default_value=0.0, theta_default_value=0.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=3.0, y_default_value=3.0, theta_default_value=0.0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    progress_1 = progress.evaluate(context=evaluation_context_1)
    logger.info(f"Test progress along reference results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Progress along reference: {progress_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    # P3: check all progresses are smaller than 0
    assert 0.0 > progress_0[P1].value
    assert 0.0 > progress_0[P2].value
    assert 0.0 > progress_0[P3].value

    # scenario 1 vs scenario 0
    # P1: offset in positive y should improve (i.e. decrease) progress
    # P2: offset in positive x should improve (i.e. decrease) progress
    # P3: offset in positive x and positive y should worsen (i.e. increase) progress
    assert progress_0[P1].value > progress_1[P1].value
    assert progress_0[P2].value > progress_1[P2].value
    assert progress_1[P3].value > progress_0[P3].value

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_default_value=2.0, y_default_value=0.0, theta_default_value=0.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_default_value=0.0, y_default_value=2.0, theta_default_value=0.0),
        P3: _PlayerOffsets(
            size=size_p3_trajectory, x_default_value=-3.0, y_default_value=-3.0, theta_default_value=0.0
        ),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    progress_2 = progress.evaluate(context=evaluation_context_2)
    logger.info(f"Test progress along reference results:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Progress along reference: {progress_2}")
    logger.info(f"--------------------------------------------\n")

    # scenario 2
    # P1: offset in x should have small impact on progress
    # P2: offset in y should have small impact on progress
    # P3: offset in negative x and negative y should improve (i.e. decrease) progress
    assert abs(progress_2[P1].value - progress_0[P1].value) < 1.0
    assert abs(progress_2[P2].value - progress_0[P2].value) < 1.0
    assert progress_0[P3].value > progress_2[P3].value

    # scenario 2 vs scenario 0 and scenario 2 vs scenario 1
    # P1: offset in positive y should improve (i.e. decrease) progress
    # P2: offset in positive x should improve (i.e. decrease) progress
    # P3: offset in negative x and negative y should improve (i.e. decrease) progress
    assert progress_2[P1].value > progress_1[P1].value
    assert progress_2[P2].value > progress_1[P2].value
    assert progress_0[P3].value > progress_2[P3].value
    assert progress_1[P3].value > progress_2[P3].value

    joint_player_offsets_3 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory, x_default_value=0.0, y_default_value=0.0, theta_default_value=pi / 3
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory,
            x_default_value=0.0,
            y_default_value=0.0,
            theta_default_value=0.0,
            v_default_value=10,
        ),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=5.0, y_default_value=5.0, theta_default_value=0.0),
    }

    evaluation_context_3 = get_default_evaluation_context(joint_player_offsets_3)
    visualize_evaluation_context(context=evaluation_context_3, show_plot=show_plots)
    progress_3 = progress.evaluate(context=evaluation_context_3)
    logger.info(f"Test progress along reference results:")
    logger.info(joint_player_offsets_3)
    logger.info(f"Progress along reference: {progress_3}")
    logger.info(f"--------------------------------------------\n")

    # scenario 3 vs scenario 0
    # P1: offset in theta should not change progress
    # P2: offset in velocity should not change progress
    # scenario 3 vs scenario 1
    # P3: larger offset in positive x and positive y should worsen progress (i.e. increase)
    assert abs(progress_3[P1].value - progress_0[P1].value) < 0.01
    assert abs(progress_3[P2].value - progress_0[P2].value) < 0.01
    assert progress_3[P3].value > progress_1[P3].value

    logger.info(f"Test Progress Along Reference finished.")


def test_longitudinal_acceleration():
    show_plots = False
    longitudinal_acceleration = LongitudinalAcceleration()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    long_acc_0 = longitudinal_acceleration.evaluate(context=evaluation_context)
    logger.info(f"Test longitudinal acceleration results:")
    logger.info(f"No Offset.")
    logger.info(f"Longitudinal acceleration: {long_acc_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory,
            x_default_value=0.0,
            y_default_value=2.0,
            theta_default_value=0.0,
            acc_default_value=1.0,
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory,
            x_default_value=0.0,
            y_default_value=0.0,
            theta_default_value=0.0,
            acc_default_value=-1.0,
        ),
        P3: _PlayerOffsets(
            size=size_p3_trajectory,
            x_default_value=0.0,
            y_default_value=3.0,
            theta_default_value=0.0,
            acc_default_value=1.0,
        ),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    long_acc_1 = longitudinal_acceleration.evaluate(context=evaluation_context_1)
    logger.info(f"Test longitudinal acceleration results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Longitudinal acceleration: {long_acc_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    # P1: longitudinal acceleration should be zero
    # P2: longitudinal acceleration should be zero
    # P3: longitudinal acceleration should be zero
    assert long_acc_0[P1].value == 0.0
    assert long_acc_0[P2].value == 0.0
    assert long_acc_0[P3].value == 0.0

    # scenario 1 vs scenario 0
    # P1: longitudinal acceleration should be greater than 0
    # P2: longitudinal acceleration should smaller than 0
    # P3: longitudinal acceleration should greater than 0
    assert long_acc_1[P1].value > long_acc_0[P1].value
    assert long_acc_0[P2].value > long_acc_1[P2].value
    assert long_acc_1[P3].value > long_acc_0[P3].value

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(
            size=size_p1_trajectory,
            x_default_value=0.0,
            y_default_value=2.0,
            theta_default_value=0.0,
            acc_default_value=2.0,
        ),
        P2: _PlayerOffsets(
            size=size_p2_trajectory,
            x_default_value=0.0,
            y_default_value=0.0,
            theta_default_value=0.0,
            acc_default_value=-2.0,
        ),
        P3: _PlayerOffsets(
            size=size_p3_trajectory,
            x_default_value=0.0,
            y_default_value=3.0,
            theta_default_value=0.0,
            acc_default_value=2.0,
        ),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    long_acc_2 = longitudinal_acceleration.evaluate(context=evaluation_context_2)
    logger.info(f"Test longitudinal acceleration results:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Longitudinal acceleration: {long_acc_2}")
    logger.info(f"--------------------------------------------\n")

    # scenario 2 vs scenario 1
    # P1: longitudinal acceleration should be greater in scenario 2
    # P2: longitudinal acceleration should smaller in scenario 2
    # P3: longitudinal acceleration should greater in scenario 2
    assert long_acc_2[P1].value > long_acc_1[P1].value
    assert long_acc_1[P2].value > long_acc_2[P2].value
    assert long_acc_2[P3].value > long_acc_1[P3].value

    logger.info(f"Test Longitudinal Acceleration finished.")


def test_lateral_comfort():
    show_plots = False
    lateral_comfort = LateralComfort()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    lat_comf_0 = lateral_comfort.evaluate(context=evaluation_context)
    logger.info(f"Test lateral comfort results:")
    logger.info(f"No Offset.")
    logger.info(f"Lateral comfort: {lat_comf_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, v_default_value=2.0, delta_default_value=-pi / 6),
        P2: _PlayerOffsets(size=size_p2_trajectory, v_default_value=2.0, delta_default_value=pi / 3),
        P3: _PlayerOffsets(size=size_p3_trajectory, v_default_value=2.0, delta_default_value=pi / 2),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    lat_comf_1 = lateral_comfort.evaluate(context=evaluation_context_1)
    logger.info(f"Test lateral comfort results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Lateral comfort: {lat_comf_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, v_default_value=4.0, delta_default_value=-pi / 6),
        P2: _PlayerOffsets(size=size_p2_trajectory, v_default_value=2.0, delta_default_value=pi / 2),
        P3: _PlayerOffsets(size=size_p3_trajectory, v_default_value=2.0, delta_default_value=-pi / 2),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    lat_comf_2 = lateral_comfort.evaluate(context=evaluation_context_2)
    logger.info(f"Test lateral comfort results:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Lateral comfort: {lat_comf_2}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    # since velociy is 0 for all players, lateral comfort metric should also be zero
    assert lat_comf_0[P1].value == 0.0
    assert lat_comf_0[P2].value == 0.0
    assert lat_comf_0[P3].value == 0.0

    # scenario 1 vs scenario 0
    # all lateral comforts metrics should increase. P3>P2>P1
    assert lat_comf_1[P1].value > 0.0
    assert lat_comf_1[P2].value > lat_comf_1[P1].value
    assert lat_comf_1[P3].value > lat_comf_1[P2].value

    # scenario 2 vs scenario 1
    # increasing either longitudinal velocity or steering angle should increase metric
    # changing sign of steering angle should not change value
    assert lat_comf_2[P1].value > lat_comf_1[P1].value
    assert lat_comf_2[P2].value > lat_comf_1[P2].value
    assert lat_comf_2[P3].value == lat_comf_1[P3].value

    logger.info(f"Test Lateral Comfort finished.")


def test_steering_angle():
    show_plots = False
    steering_angle = SteeringAngle()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    steering_angle_0 = steering_angle.evaluate(context=evaluation_context)
    logger.info(f"Test steering angle results:")
    logger.info(f"No Offset.")
    logger.info(f"Steering Angle: {steering_angle_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, delta_default_value=-pi / 6),
        P2: _PlayerOffsets(size=size_p2_trajectory, delta_default_value=pi / 3),
        P3: _PlayerOffsets(size=size_p3_trajectory, delta_default_value=-pi / 2),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    steering_angle_1 = steering_angle.evaluate(context=evaluation_context_1)
    logger.info(f"Test steering angle results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Lateral comfort: {steering_angle_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    assert steering_angle_0[P1].value == 0
    assert steering_angle_0[P2].value == 0
    assert steering_angle_0[P3].value == 0

    # scenario 1
    # all values should be greater than 0. Value should increase if abs(angle) increases
    assert steering_angle_1[P1].value > steering_angle_0[P1].value
    assert steering_angle_1[P2].value > steering_angle_1[P1].value
    assert steering_angle_1[P3].value > steering_angle_1[P2].value

    logger.info(f"Test Steering Angle finished.")


def test_steering_rate():
    show_plots = False
    steering_rate = SteeringRate()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    steering_rate_0 = steering_rate.evaluate(context=evaluation_context)
    logger.info(f"Test steering angle results:")
    logger.info(f"No Offset.")
    logger.info(f"Steering Angle: {steering_rate_0}")
    logger.info(f"--------------------------------------------\n")

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, delta_rate_default_value=1.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, delta_rate_default_value=2.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, delta_rate_default_value=3.0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    steering_rate_1 = steering_rate.evaluate(context=evaluation_context_1)
    logger.info(f"Test steering angle results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Lateral comfort: {steering_rate_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    assert steering_rate_0[P1].value == 0
    assert steering_rate_0[P2].value == 0
    assert steering_rate_0[P3].value == 0

    # scenario 1
    # all values should be greater than 0. Value should increase if abs(angle) increases
    assert steering_rate_1[P1].value > steering_rate_0[P1].value
    assert steering_rate_1[P2].value > steering_rate_1[P1].value
    assert steering_rate_1[P3].value > steering_rate_1[P2].value

    logger.info(f"Test Steering Angle finished.")


def test_clearance():
    show_plots = False
    clearance = Clearance()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    clearance_0 = clearance.evaluate(context=evaluation_context)
    logger.info(f"Test clearance results:")
    logger.info(f"No Offset.")
    logger.info(f"Clearance: {clearance_0}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0
    assert clearance_0[P1].value > 0
    assert clearance_0[P2].value > 0
    assert clearance_0[P3].value > 0

    logger.info(f"Test clearance finished.")


def test_collision_energy():
    show_plots = False
    coll_energy = CollisionEnergy()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    coll_energy_0 = coll_energy.evaluate(context=evaluation_context)
    logger.info(f"Test collision energy results:")
    logger.info(f"No Offset.")
    logger.info(f"Collision Energy: {coll_energy_0}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0: velocity is zero so collision energy should be zero
    assert coll_energy_0[P1].value == 0
    assert coll_energy_0[P2].value == 0
    assert coll_energy_0[P3].value == 0

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, v_default_value=2.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, v_default_value=2.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, v_default_value=2.0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    coll_energy_1 = coll_energy.evaluate(context=evaluation_context_1)
    logger.info(f"Test collision energy results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Collision Energy: {coll_energy_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 1 vs scenario 0
    # in scenario 1, velocity is greater and therefore collision energy should be greater than zero,
    # since in this scenario there is a collision taking place.
    # P3 should not have any collision
    assert coll_energy_1[P1].value > coll_energy_0[P1].value
    assert coll_energy_1[P2].value > coll_energy_0[P2].value
    assert coll_energy_0[P3].value == 0
    # check that cost is symmetric
    assert coll_energy_1[P1].value == coll_energy_1[P2].value

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, y_default_value=100.0, v_default_value=2.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, v_default_value=2.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, v_default_value=2.0),
    }

    evaluation_context_2 = get_default_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)
    coll_energy_2 = coll_energy.evaluate(context=evaluation_context_2)
    logger.info(f"Test collision energy results:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Collision Energy: {coll_energy_2}")
    logger.info(f"--------------------------------------------\n")

    # scenario 2
    # by shifting the trajectory of P1 in y direction, there should be no more collision.
    assert coll_energy_0[P1].value == 0
    assert coll_energy_0[P2].value == 0
    assert coll_energy_0[P3].value == 0

    logger.info(f"Test collision finished.")


def test_minimum_clearance():
    show_plots = False
    min_clearance = MinimumClearance()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    min_clearance_0 = min_clearance.evaluate(context=evaluation_context)
    logger.info(f"Test minimum clearance results:")
    logger.info(f"No Offset.")
    logger.info(f"Minimum clearance: {min_clearance_0}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0: clearance is always greater than 0
    assert min_clearance_0[P1].value > 0
    assert min_clearance_0[P2].value > 0
    assert min_clearance_0[P3].value > 0

    min_clearance_new = MinimumClearance()
    min_clearance_new.min_clearance = 3.0
    min_clearance_1 = min_clearance_new.evaluate(context=evaluation_context)
    logger.info(f"Test minimum clearance results:")
    logger.info(f"No Offset, but new minimum clearance threshold: {3.0}")
    logger.info(f"Minimum Clearance: {min_clearance_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 1 vs scenario 0: decreasing clearance threshold from 10.0 (default) to 5.0 should decrease metric
    assert min_clearance_0[P1].value > min_clearance_1[P1].value
    assert min_clearance_0[P2].value > min_clearance_1[P2].value
    assert min_clearance_0[P3].value > min_clearance_1[P3].value

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory),
        P2: _PlayerOffsets(size=size_p2_trajectory),
        P3: _PlayerOffsets(size=size_p3_trajectory, x_default_value=5.0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    min_clearance_2 = min_clearance_new.evaluate(context=evaluation_context_1)
    logger.info(f"Test minimum clearance results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Minimum clearance: {min_clearance_2}")
    logger.info(f"--------------------------------------------\n")

    # scenario 2 vs scenario 1: shifting trajectory of P3 in positive x direction should increase clearance violation
    assert min_clearance_2[P3].value > min_clearance_1[P3].value
    assert min_clearance_2[P1].value > min_clearance_1[P1].value


def test_clearance_time_violation():
    show_plots = False
    clear_viol_time = ClearanceViolationTime()
    evaluation_context = get_default_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)
    viol_time_0 = clear_viol_time.evaluate(context=evaluation_context)
    logger.info(f"Test clearance violation time results:")
    logger.info(f"No Offset.")
    logger.info(f"Minimum Clearance Violation Time: {viol_time_0}")
    logger.info(f"--------------------------------------------\n")

    # scenario 0: check that all metrics are greater than 0
    assert viol_time_0[P1].value > 0
    assert viol_time_0[P2].value > 0
    assert viol_time_0[P3].value > 0

    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, y_default_value=100.0),
        P2: _PlayerOffsets(size=size_p2_trajectory, v_default_value=5.0),
        P3: _PlayerOffsets(size=size_p3_trajectory, y_default_value=5.0),
    }

    evaluation_context_1 = get_default_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)
    viol_time_1 = clear_viol_time.evaluate(context=evaluation_context_1)
    logger.info(f"Test clearance violation time results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Minimum Clearance Violation Time: {viol_time_1}")
    logger.info(f"--------------------------------------------\n")

    # scenario 1 vs scenario 0: moving red trajectory out of the way
    # should decrease violation times for the other trajectories
    assert viol_time_1[P1].value == 0
    assert viol_time_0[P2].value > viol_time_1[P2].value
    assert viol_time_0[P3].value > viol_time_1[P3].value


def get_goal_violation_evaluation_context(
        player_offsets: Optional[JointPlayerOffsets] = None) -> MetricEvaluationContext:
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    dgscenario = DgScenario(scenario)

    # defines reference lanelets for players
    lanelet_network = scenario.lanelet_network

    points_from_first = 4
    points_from_last = 4

    lane1_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49598))
    lane3_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49576))

    lane1_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49588))
    lane3_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49566))

    lane1_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49580))
    lane3_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49572))

    north_ctrl_points = (
            lane1_north.control_points[-points_from_last:-1]
            + lane2_north.control_points
            + lane3_north.control_points[1:points_from_first]
    )

    west_ctrl_points = (
            lane1_west.control_points[-points_from_last:-1]
            + lane2_west.control_points
            + lane3_west.control_points[1:points_from_first]
    )

    east_ctrl_points = (
            lane1_east.control_points[-points_from_last:-1]
            + lane2_east.control_points
            + lane3_east.control_points[1:points_from_first]
    )

    dglanelet_north = DgLanelet(north_ctrl_points)
    dglanelet_west = DgLanelet(west_ctrl_points)
    dglanelet_east = DgLanelet(east_ctrl_points)

    # we test P1's goal against other available goals
    # second and third alternative are equivalent
    goals = {
        P1: [RefLaneGoal(ref_lane=dglanelet_north, goal_progress=0.8),  # P1's real goal
             RefLaneGoal(ref_lane=dglanelet_west, goal_progress=0.8),  # first alternative
             RefLaneGoal(ref_lane=dglanelet_east, goal_progress=0.8)],  # second alternative
        P2: [RefLaneGoal(ref_lane=dglanelet_west, goal_progress=0.8),  # P2's real goal
             RefLaneGoal(ref_lane=dglanelet_east, goal_progress=0.8),  # first alternative
             RefLaneGoal(ref_lane=dglanelet_north, goal_progress=0.8)],  # second alternative
        P3: [RefLaneGoal(ref_lane=dglanelet_east, goal_progress=0.8),  # P3's real goal
             RefLaneGoal(ref_lane=dglanelet_west, goal_progress=0.8),  # first alternative
             RefLaneGoal(ref_lane=dglanelet_north, goal_progress=0.8)],  # second alternative
    }

    # Define trajectories for players
    x_1_translation_angles = [
        translation_angle_from_SE2(dglanelet_north.center_point(beta)) for beta in
        range(len(dglanelet_north.control_points))
    ]
    x_2_translation_angles = [
        translation_angle_from_SE2(dglanelet_west.center_point(beta)) for beta in
        range(len(dglanelet_west.control_points))
    ]
    x_3_translation_angles = [
        translation_angle_from_SE2(dglanelet_east.center_point(beta)) for beta in
        range(len(dglanelet_east.control_points))
    ]

    global size_p1_trajectory, size_p2_trajectory, size_p3_trajectory

    size_p1_trajectory = len(x_1_translation_angles)
    size_p2_trajectory = len(x_2_translation_angles)
    size_p3_trajectory = len(x_3_translation_angles)

    if player_offsets is None:
        player_offsets = {
            P1: _PlayerOffsets(size=size_p1_trajectory),
            P2: _PlayerOffsets(size=size_p2_trajectory),
            P3: _PlayerOffsets(size=size_p3_trajectory),
        }

    x_1 = [
        VehicleState(
            x=translation[0] + player_offsets[P1].x_offset[i],
            y=translation[1] + player_offsets[P1].y_offset[i],
            theta=angle + player_offsets[P1].theta_offset[i],
            vx=0.0 + player_offsets[P1].v_offset[i],
            delta=0.0 + player_offsets[P1].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_1_translation_angles)
    ]

    x_2 = [
        VehicleState(
            x=translation[0] + player_offsets[P2].x_offset[i],
            y=translation[1] + player_offsets[P2].y_offset[i],
            theta=angle + player_offsets[P2].theta_offset[i],
            vx=0.0 + player_offsets[P2].v_offset[i],
            delta=0.0 + player_offsets[P2].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_2_translation_angles)
    ]

    x_3 = [
        VehicleState(
            x=translation[0] + player_offsets[P3].x_offset[i],
            y=translation[1] + player_offsets[P3].y_offset[i],
            theta=angle + player_offsets[P3].theta_offset[i],
            vx=0.0 + player_offsets[P3].v_offset[i],
            delta=0.0 + player_offsets[P3].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_3_translation_angles)
    ]

    t_max = 10.0
    joint_trajectories: JointTrajectories = {
        P1: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_1))), values=x_1),
        P2: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_2))), values=x_2),
        P3: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_3))), values=x_3),
    }

    geos = {
        P1: VehicleGeometry.default_car(),
        P2: VehicleGeometry.default_car(),
        P3: VehicleGeometry.default_car(),
    }

    return MetricEvaluationContext(dgscenario=dgscenario, trajectories=joint_trajectories, goals=goals, geos=geos)


def test_goal_violation():
    show_plots = False
    goal_violation = GoalViolation()
    evaluation_context = get_goal_violation_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)

    goal_viol_0 = goal_violation.evaluate(context=evaluation_context)
    logger.info(f"Test goal violation results:")
    logger.info(f"No Offset.")
    logger.info(f"Goal Violation: {goal_viol_0}")
    logger.info(f"--------------------------------------------\n")

    # all trajectories are equal to the reference lanes -> metric should evaluate to 0
    assert goal_viol_0[P1].value == 0.0
    assert goal_viol_0[P2].value == 0.0
    assert goal_viol_0[P3].value == 0.0

    x_offset_p1_1 = list(np.linspace(-10.0, 10.0, size_p1_trajectory))
    joint_player_offsets_1 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_offset=x_offset_p1_1),
        P2: _PlayerOffsets(size=size_p2_trajectory),
        P3: _PlayerOffsets(size=size_p3_trajectory),
    }


    evaluation_context_1 = get_goal_violation_evaluation_context(joint_player_offsets_1)
    visualize_evaluation_context(context=evaluation_context_1, show_plot=show_plots)



    goal_viol_1 = goal_violation.evaluate(context=evaluation_context_1)
    logger.info(f"Test goal violation results:")
    logger.info(joint_player_offsets_1)
    logger.info(f"Goal Violation: {goal_viol_1}")
    logger.info(f"--------------------------------------------\n")

    # P1's trajectory is rotated but still most similar to P1's reference. Metric should evaluate to 0
    assert goal_viol_1[P1].value == 0.0
    assert goal_viol_1[P2].value == 0.0
    assert goal_viol_1[P3].value == 0.0



    x_offset_p1_2 = list(np.linspace(0.0, 100.0, int(size_p1_trajectory / 2)))
    x_offset_p1_2 = [0 for i in range(int(size_p1_trajectory / 2))] + x_offset_p1_2
    y_offset_p1_2 = list(np.linspace(0.0, -100.0, int(size_p1_trajectory / 2)))
    y_offset_p1_2 = [0 for i in range(int(size_p1_trajectory / 2))] + y_offset_p1_2

    joint_player_offsets_2 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_offset=x_offset_p1_2, y_offset=y_offset_p1_2),
        P2: _PlayerOffsets(size=size_p2_trajectory),
        P3: _PlayerOffsets(size=size_p3_trajectory),
    }


    evaluation_context_2 = get_goal_violation_evaluation_context(joint_player_offsets_2)
    visualize_evaluation_context(context=evaluation_context_2, show_plot=show_plots)

    goal_viol_2 = goal_violation.evaluate(context=evaluation_context_2)
    logger.info(f"Test goal violation results:")
    logger.info(joint_player_offsets_2)
    logger.info(f"Goal Violation: {goal_viol_2}")
    logger.info(f"--------------------------------------------\n")

    # P1's trajectory is now closet to P3's reference. Metric for P1 should be greater than 0
    assert goal_viol_2[P1].value > 0.0
    assert goal_viol_2[P2].value == 0.0
    assert goal_viol_2[P3].value == 0.0

    x_offset_p1_3 = list(np.linspace(0.0, -100.0, int(size_p1_trajectory / 2)))
    x_offset_p1_3 = [0 for i in range(int(size_p1_trajectory / 2))] + x_offset_p1_3
    y_offset_p1_3 = list(np.linspace(0.0, -100.0, int(size_p1_trajectory / 2)))
    y_offset_p1_3 = [0 for i in range(int(size_p1_trajectory / 2))] + y_offset_p1_3

    joint_player_offsets_3 = {
        P1: _PlayerOffsets(size=size_p1_trajectory, x_offset=x_offset_p1_3, y_offset=y_offset_p1_3),
        P2: _PlayerOffsets(size=size_p2_trajectory),
        P3: _PlayerOffsets(size=size_p3_trajectory),
    }

    evaluation_context_3 = get_goal_violation_evaluation_context(joint_player_offsets_3)
    visualize_evaluation_context(context=evaluation_context_3, show_plot=show_plots)

    goal_viol_3 = goal_violation.evaluate(context=evaluation_context_3)
    logger.info(f"Test goal violation results:")
    logger.info(joint_player_offsets_3)
    logger.info(f"Goal Violation: {goal_viol_3}")
    logger.info(f"--------------------------------------------\n")

    # P1's trajectory is now closet to P2's reference. Metric for P1 should be greater than 0
    assert goal_viol_3[P1].value > 0.0
    assert goal_viol_3[P2].value == 0.0
    assert goal_viol_3[P3].value == 0.0

    x_offset_p2_1 = list(np.linspace(0, 125.0, int(size_p2_trajectory / 2)))
    x_offset_p2_1 = [0 for i in range(int(1 + size_p2_trajectory / 2))] + x_offset_p2_1
    y_offset_p2_1 = list(np.linspace(0.0, 0.0, int(size_p2_trajectory / 2)))
    y_offset_p2_1 = [0 for i in range(int(1 + size_p2_trajectory / 2))] + y_offset_p2_1

    joint_player_offsets_4 = {
        P1: _PlayerOffsets(size=size_p1_trajectory),
        P2: _PlayerOffsets(size=size_p2_trajectory, x_offset=x_offset_p2_1, y_offset=y_offset_p2_1),
        P3: _PlayerOffsets(size=size_p3_trajectory),
    }

    evaluation_context_4 = get_goal_violation_evaluation_context(joint_player_offsets_4)
    visualize_evaluation_context(context=evaluation_context_4, show_plot=show_plots)

    goal_viol_4 = goal_violation.evaluate(context=evaluation_context_4)
    logger.info(f"Test goal violation results:")
    logger.info(joint_player_offsets_4)
    logger.info(f"Goal Violation: {goal_viol_4}")
    logger.info(f"--------------------------------------------\n")

    # P2's trajectory is now closet to P3's reference. Metric for P2 should be greater than 0
    assert goal_viol_4[P1].value == 0.0
    assert goal_viol_4[P2].value > 0.0
    assert goal_viol_4[P3].value == 0.0


def get_traffic_rules_evaluation_context(
        player_offsets: Optional[JointPlayerOffsets] = None) -> MetricEvaluationContext:
    SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")
    scenario, _ = load_commonroad_scenario("DEU_Ffb-1_7_T-1", SCENARIOS_DIR)
    scenario = add_traffic_light_custom(scenario=scenario)

    dgscenario = DgScenario(scenario)

    # defines reference lanelets for players
    lanelet_network = scenario.lanelet_network

    points_from_first = 4
    points_from_last = 4

    lane1_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49598))
    lane3_north = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49576))

    lane1_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49588))
    lane3_west = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49566))

    lane1_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49570))
    lane2_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49580))
    lane3_east = DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(49572))

    north_ctrl_points = (
            lane1_north.control_points[-points_from_last:-1]
            + lane2_north.control_points
            + lane3_north.control_points[1:points_from_first]
    )

    west_ctrl_points = (
            lane1_west.control_points[-points_from_last:-1]
            + lane2_west.control_points
            + lane3_west.control_points[1:points_from_first]
    )

    east_ctrl_points = (
            lane1_east.control_points[-points_from_last:-1]
            + lane2_east.control_points
            + lane3_east.control_points[1:points_from_first]
    )

    dglanelet_north = DgLanelet(north_ctrl_points)
    dglanelet_west = DgLanelet(west_ctrl_points)
    dglanelet_east = DgLanelet(east_ctrl_points)


    goals = {
        P1: [RefLaneGoal(ref_lane=dglanelet_north, goal_progress=0.8)],
        P2: [RefLaneGoal(ref_lane=dglanelet_west, goal_progress=0.8)],
        P3: [RefLaneGoal(ref_lane=dglanelet_east, goal_progress=0.8)],
    }

    # Define trajectories for players
    x_1_translation_angles = [
        translation_angle_from_SE2(dglanelet_north.center_point(beta)) for beta in
        range(len(dglanelet_north.control_points))
    ]
    x_2_translation_angles = [
        translation_angle_from_SE2(dglanelet_west.center_point(beta)) for beta in
        range(len(dglanelet_west.control_points))
    ]
    x_3_translation_angles = [
        translation_angle_from_SE2(dglanelet_east.center_point(beta)) for beta in
        range(len(dglanelet_east.control_points))
    ]

    global size_p1_trajectory, size_p2_trajectory, size_p3_trajectory

    size_p1_trajectory = len(x_1_translation_angles)
    size_p2_trajectory = len(x_2_translation_angles)
    size_p3_trajectory = len(x_3_translation_angles)

    if player_offsets is None:
        player_offsets = {
            P1: _PlayerOffsets(size=size_p1_trajectory),
            P2: _PlayerOffsets(size=size_p2_trajectory),
            P3: _PlayerOffsets(size=size_p3_trajectory),
        }

    x_1 = [
        VehicleState(
            x=translation[0] + player_offsets[P1].x_offset[i],
            y=translation[1] + player_offsets[P1].y_offset[i],
            theta=angle + player_offsets[P1].theta_offset[i],
            vx=0.0 + player_offsets[P1].v_offset[i],
            delta=0.0 + player_offsets[P1].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_1_translation_angles)
    ]

    x_2 = [
        VehicleState(
            x=translation[0] + player_offsets[P2].x_offset[i],
            y=translation[1] + player_offsets[P2].y_offset[i],
            theta=angle + player_offsets[P2].theta_offset[i],
            vx=0.0 + player_offsets[P2].v_offset[i],
            delta=0.0 + player_offsets[P2].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_2_translation_angles)
    ]

    x_3 = [
        VehicleState(
            x=translation[0] + player_offsets[P3].x_offset[i],
            y=translation[1] + player_offsets[P3].y_offset[i],
            theta=angle + player_offsets[P3].theta_offset[i],
            vx=0.0 + player_offsets[P3].v_offset[i],
            delta=0.0 + player_offsets[P3].delta_offset[i],
        )
        for i, (translation, angle) in enumerate(x_3_translation_angles)
    ]

    t_max = 10.0
    joint_trajectories: JointTrajectories = {
        P1: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_1))), values=x_1),
        P2: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_2))), values=x_2),
        P3: Trajectory(timestamps=list(np.linspace(0, t_max, num=len(x_3))), values=x_3),
    }

    geos = {
        P1: VehicleGeometry.default_car(),
        P2: VehicleGeometry.default_car(),
        P3: VehicleGeometry.default_car(),
    }

    return MetricEvaluationContext(dgscenario=dgscenario, trajectories=joint_trajectories, goals=goals, geos=geos)

def test_traffic_lights_violation():
    show_plots = True
    evaluation_context = get_traffic_rules_evaluation_context()
    visualize_evaluation_context(context=evaluation_context, show_plot=show_plots)

if __name__ == "__main__":
    matplotlib.use("TkAgg")
    # test_times()
    # test_lateral_deviation()
    # test_heading_deviation()
    # test_drivable_area_violation()
    # test_progress_along_reference()
    # test_longitudinal_acceleration()
    # test_lateral_comfort()
    # test_steering_angle()
    # test_steering_rate()
    # test_clearance()
    # test_collision_energy()
    # test_minimum_clearance()
    # test_clearance_time_violation()
    # test_goal_violation()
    test_traffic_lights_violation()
