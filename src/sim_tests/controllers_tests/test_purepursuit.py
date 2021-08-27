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
from crash.reports import generete_report
import os
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from games import PlayerName
from typing import Optional


def create_lanelet_from_scratch(dyn_obs, width):
    center_points = []
    left = []
    right = []
    for state in dyn_obs.prediction.trajectory.state_list:
        center_points.append(state.position)

    center_points = np.array(center_points)

    delta = width*(center_points[1]-center_points[0])
    left.append(np.array([-delta[1], delta[0]]))
    right.append(np.array([delta[1], -delta[0]]))
    for i, _ in enumerate(center_points):
        if i != 0 and i != (len(center_points)-1):
            delta = width * (center_points[i-1] - center_points[i+1])
            left.append(np.array([-delta[1], delta[0]]))
            right.append(np.array([delta[1], -delta[0]]))

    delta = width * (center_points[-1] - center_points[-2])
    left.append(np.array([-delta[1], delta[0]]))
    right.append(np.array([delta[1], -delta[0]]))

    left = np.array(left)
    right = np.array(right)

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
        width = np.average(np.linalg.norm(lanelet.right_vertices - lanelet.left_vertices))
        return_lanelet = create_lanelet_from_scratch(dyn_obs, width)

    return return_lanelet


def lp_agent_from_dynamic_obstacle(dyn_obs: DynamicObstacle, lanes, nominal_speed: float,
                                   ddelta_kp: float, k_lookahead: float) -> (LFAgent, VehicleModel):
    assert dyn_obs.obstacle_type == ObstacleType.CAR

    x0 = VehicleState(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                      theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity, delta=0)
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

    report = generete_report(sim_context)
    # save report
    output_dir = "out"
    report_file = os.path.join(output_dir, f"Pure_Pursuit_test_speed{vehicle_speed}_lookahead{k_lookahead}_ddeltakp{ddelta_kp}.html")
    report.to_html(report_file)


test_pure_pursuit_control_all_vehicles()
