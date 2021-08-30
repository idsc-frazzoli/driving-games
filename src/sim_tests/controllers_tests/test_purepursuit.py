import math
import matplotlib.pyplot as plt
from commonroad.scenario.lanelet import Lanelet
from dg_commons.planning.lanes import DgLanelet, LaneCtrPoint
from dg_commons.controllers.speed import SpeedBehavior
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from sim.scenarios import load_commonroad_scenario
from sim.agents.lane_follower import LFAgent
from sim.simulator import SimContext, Simulator, SimParameters, SimulationLog
from sim.models.vehicle import VehicleModel, VehicleState
import numpy as np
from crash.reports import generate_report
import os
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from games import PlayerName
from typing import Optional
from geometry import translation_angle_from_SE2


def compute_lanelet_given_trajectory(center_points, width):
    left = []
    right = []

    def update_left_right(point_1, point_2):
        delta = point_2 - point_1
        delta /= np.linalg.norm(delta)
        delta *= width/2
        left.append(np.array([-delta[1], delta[0]]))
        right.append(np.array([delta[1], -delta[0]]))
        return delta

    update_left_right(center_points[0], center_points[1])
    for i, _ in enumerate(center_points):
        if i != 0 and i != (len(center_points)-1):
            update_left_right(center_points[i-1], center_points[i+1])

    update_left_right(center_points[-2], center_points[-1])

    return np.array(left), np.array(right)


def get_additional_trajectory(lanes, end_position, width):
    current_lanelet_id = lanes.find_lanelet_by_position([end_position])[0][0]
    current_lanelet = lanes.find_lanelet_by_id(current_lanelet_id)
    merged_lanelets, _ = current_lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet, lanes)
    merged_lanelet = merged_lanelets[0]
    center_points = merged_lanelet.center_vertices.tolist()

    dg_lanelet = DgLanelet.from_commonroad_lanelet(merged_lanelet)
    beta, _ = dg_lanelet.find_along_lane_closest_point(end_position)
    first_center, _ = translation_angle_from_SE2(dg_lanelet.center_point(beta))

    center_points = center_points[math.ceil(beta):]
    center_points = [first_center.tolist()] + center_points
    center_points = np.array(center_points)

    left, right = compute_lanelet_given_trajectory(center_points, width)
    return center_points, left, right


def create_lanelet_from_scratch(lanes, dyn_obs, width):
    center_points = []
    for state in dyn_obs.prediction.trajectory.state_list:
        center_points.append(state.position)

    center_points = np.array(center_points)
    left, right = compute_lanelet_given_trajectory(center_points, width)
    add_center, add_left, add_right = get_additional_trajectory(lanes, center_points[-1], width)

    center_points = np.concatenate((center_points, add_center))
    left = np.concatenate((left, add_left))
    right = np.concatenate((right, add_right))

    return Lanelet(left_vertices=left, right_vertices=right, center_vertices=center_points, lanelet_id=123456)


def get_merged_lanes(lanes, dyn_obs):
    initial_state = dyn_obs.initial_state.position
    end_state = dyn_obs.prediction.trajectory.state_list[-1].position

    initial_ids = lanes.find_lanelet_by_position([initial_state])[0]
    final_ids = lanes.find_lanelet_by_position([end_state])[0]
    return_lanelet: Optional[Lanelet] = None

    for initial_id in initial_ids:
        lanelet = lanes.find_lanelet_by_id(initial_id)
        merged_lanelets, merged_ids = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanes)
        for i, merged_id in enumerate(merged_ids):
            if any(item in merged_id for item in final_ids):
                return_lanelet = merged_lanelets[i]
                break
        if return_lanelet is not None:
            break

    if return_lanelet is None:
        lanelet = lanes.find_lanelet_by_id(initial_ids[0])
        width = np.average(np.linalg.norm(lanelet.right_vertices - lanelet.left_vertices, axis=1))
        return_lanelet = create_lanelet_from_scratch(lanes, dyn_obs, width)

    return return_lanelet


def lp_agent_from_dynamic_obstacle(dyn_obs: DynamicObstacle, lanes, nominal_speed: float,
                                   ddelta_kp: float, k_lookahead: float) -> (LFAgent, VehicleModel):
    assert dyn_obs.obstacle_type == ObstacleType.CAR

    orient_0, orient_1 = dyn_obs.prediction.trajectory.state_list[0].orientation, dyn_obs.prediction.trajectory.state_list[1].orientation
    vel_0 = dyn_obs.prediction.trajectory.state_list[0].velocity
    dtheta = orient_1-orient_0
    l = dyn_obs.obstacle_shape.length
    delta_0 = math.atan(l * dtheta / vel_0)

    x0 = VehicleState(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                      theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity, delta=delta_0)
    model = VehicleModel.default_car(x0=x0)

    merged_lane = get_merged_lanes(lanes, dyn_obs)
    dg_lane = DgLanelet.from_commonroad_lanelet(merged_lane)
    points = dg_lane.lane_profile(10)
    x, y = zip(*points)
    plt.fill(np.array(x), np.array(y), color=model.vg.color, alpha=0.2)

    pure_pursuit_param: PurePursuitParam = PurePursuitParam(k_lookahead=k_lookahead)
    pure_pursuit_controller: PurePursuit = PurePursuit(pure_pursuit_param)
    speed_behavior: SpeedBehavior = SpeedBehavior()
    speed_behavior.params.nominal_speed = nominal_speed

    agent: LFAgent = LFAgent(dg_lane, speed_behavior=speed_behavior,
                             pure_pursuit=pure_pursuit_controller, ddelta_kp=ddelta_kp)
    return agent, model


def get_sim_context_all_vehicles_controller(scenario_name: str, nominal_speed: float, ddelta_kp: float, k_lookahead: float) -> SimContext:
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    lanes = scenario.lanelet_network
    players, models = {}, {}
    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        agent, model = lp_agent_from_dynamic_obstacle(dyn_obs, lanes, nominal_speed, ddelta_kp, k_lookahead)
        player_name = PlayerName(f"P{i}")
        players.update({player_name: agent})
        models.update({player_name: model})

    plt.savefig("plotlanes.png")

    return SimContext(scenario=scenario, models=models, players=players,
                      param=SimParameters.default(), log=SimulationLog())


def test_pure_pursuit_control_all_vehicles():
    scenario_name: str = "USA_Peach-1_1_T-1"
    #scenario_name: str = "ZAM_Tjunction-1_129_T-1"
    #scenario_name: str = "ARG_Carcarana-1_1_T-1"
    """Name of the chosen scenario"""
    vehicle_speed: float = 5
    """Nominal speed of the vehicle"""
    k_lookahead: float = 1.8
    """Scaling constant for speed dependent params"""
    ddelta_kp: float = 10
    """Proportional gain ddelta with respect to delta error"""

    sim_context = get_sim_context_all_vehicles_controller(scenario_name, vehicle_speed, ddelta_kp, k_lookahead)

    simulator: Simulator = Simulator()
    simulator.run(sim_context)

    report = generate_report(sim_context)
    # save report
    output_dir = "out"
    report_file = os.path.join(output_dir, f"Pure_Pursuit_test_speed{vehicle_speed}_lookahead{k_lookahead}_ddeltakp{ddelta_kp}.html")
    report.to_html(report_file)


test_pure_pursuit_control_all_vehicles()
