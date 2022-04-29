import math
import os
from decimal import Decimal as D
from typing import List, Mapping

import matplotlib
import numpy as np
from commonroad.common.solution import VehicleType, vehicle_parameters
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from shapely.geometry import Point, LineString, Polygon
from sumocr.interface.ego_vehicle import EgoVehicle
from dg_commons.sim.models.model_structures import CAR
from dg_commons import PlayerName
from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from trajectory_games.structures import TrajectoryGenParams

goal_frac = 0.99


# find all interactive scenarios inside a scenario folder
def find_all_scenarios(scenarios_folder: str):
    subfolders = [x[0] for x in os.walk(scenarios_folder)]
    interactive_scenarios = [os.path.split(path)[1] for path in subfolders[1:]]
    return interactive_scenarios

def rectangle_around_ego(ego_state: State, look_forward_dist: float,
                         look_backward_dist: float, look_lateral_dist: float) -> Polygon:
    """
    Compute an approximate rectangle around Ego with the given sizes.
    :param ego_state: state of Ego agent
    :param look_forward_dist: Maximum longitudinal distance to look for agents (forward)
    :param look_backward_dist: Maximum longitudinal distance to look for agents (backward)
    :param look_lateral_dist: Maximum lateral distance to look for agents
    :return: Shapely Polygon
    """

    ego_pos = Point(ego_state.position)

    look_forward_pos_x = ego_pos.x + math.cos(ego_state.orientation) * look_forward_dist
    look_forward_pos_y = ego_pos.y + math.sin(ego_state.orientation) * look_forward_dist

    look_backward_pos_x = ego_pos.x - math.cos(ego_state.orientation) * look_backward_dist
    look_backward_pos_y = ego_pos.y - math.sin(ego_state.orientation) * look_backward_dist

    look_forward_point = Point(np.array([look_forward_pos_x, look_forward_pos_y]))
    look_backward_point = Point(np.array([look_backward_pos_x, look_backward_pos_y]))

    line = LineString([look_backward_point, look_forward_point])
    area_of_interest = line.buffer(distance=look_lateral_dist)
    return area_of_interest

# def leading_area_of_interest(ego_state: State, look_forward_dist: float = 40.0,
#                             lateral_search_distance: float = 1.0) -> Polygon:
#     """
#     Find area of interest to check for existence of leasing vehicles.
#     :param ego_state: state of Ego agent
#     :param look_forward_dist: Maximum longitudinal distance to look for agents
#     :param lateral_search_distance: Maximum lateral distance to look for agents
#     :return: List of agents as dynamic obstacles
#     """
#     # compute area to look for leading dynamic obstacles
#     ego_pos = Point(ego_state.position)
#     look_forward_pos_x = ego_pos.x + math.cos(ego_state.orientation)*look_forward_dist
#     look_forward_pos_y = ego_pos.y + math.sin(ego_state.orientation)*look_forward_dist
#     look_forward_point = Point(np.array([look_forward_pos_x, look_forward_pos_y]))
#
#     line = LineString([ego_pos, look_forward_point])
#     area_of_interest = line.buffer(distance=lateral_search_distance)
#     return area_of_interest




def filter_obstacles(scenario: Scenario, area_of_interest: Polygon) -> List[DynamicObstacle]:
    """
    Returns all dynamic obstacles intersecting the area of interest
    :param scenario: Commonroad Scenario
    :param area_of_interest: Shapely Polygon
    :return: Intersecting dynamic obstacles
    """

    dyn_obs = scenario.dynamic_obstacles
    inter_obs = []
    for obs in dyn_obs:
        # obs_geom = obs.obstacle_shape.shapely_object
        obs_geom = obs.occupancy_at_time(time_step=0).shape.shapely_object
        if obs_geom.intersects(area_of_interest):
            inter_obs.append(obs)

    return inter_obs

def interacting_agents(scenario: Scenario,
                       ego_state: State,
                       look_ahead_dist: float,
                       around_dist_r: float,
                       around_dist_f: float,
                       around_dist_lat: float) -> Mapping[str, List[DynamicObstacle]]:
    """

    :param scenario: Commonroad Scenario
    :param ego_state: Commonroad State of Ego Vehicle
    :param look_ahead_dist: Distance to look ahead for leading vehicles
    :param around_dist_r: Distance to look behind for surrounding vehicles
    :param around_dist_f: Distance to look forward for surrounding vehicles
    :param around_dist_lat: Distance to look laterally for surrounding vehicles
    :return: Dictionary with the closest leading vehicle and all surrounding vehicles
    """

    # find dyn. obstacles interacting with ego by being in front of it
    leading_area_of_interest = rectangle_around_ego(ego_state=ego_state,
                                                    look_forward_dist=look_ahead_dist,
                                                    look_backward_dist=0.0,
                                                    look_lateral_dist=1.5)

    leading_obs = filter_obstacles(scenario, leading_area_of_interest)



    # find dyn. obstacles interacting with ego by being close to it
    around_area_of_interest = rectangle_around_ego(ego_state=ego_state,
                                                    look_forward_dist=around_dist_f,
                                                    look_backward_dist=around_dist_r,
                                                    look_lateral_dist=around_dist_lat)

    around_obs = filter_obstacles(scenario, around_area_of_interest)

    if len(leading_obs) > 1:
        # only keep closest leading obstacle
        dist = 99999.
        closest_obs = None
        for obs in leading_obs:
            obs_dist = np.norm(obs.initial_state.position - ego_state.position) #todo: check this is correct
            if obs_dist < dist:
                closest_obs = obs
                dist = obs_dist
        leading_obs = [closest_obs]
        # remove closest leading obstacles from surrounding obstacles
        if closest_obs in around_obs:
            around_obs.remove(closest_obs)


    obs_dict: Mapping[str, List[DynamicObstacle]] = {"leading": leading_obs, "around": around_obs}
    return obs_dict

# # todo 1: probably wrong to use initial position of dynamic obstacles (?) Maybe correct. Sumo updates them (?)
# # todo: implement part with longitudinal threshold
# def interacting_lateral(ego_state: State, threshold_lateral: float = 10.0, threshold_lateral: float = 10) -> Polygon:
#     """
#     Finds agents (dynamic obstacles) that are close enough to interact with Ego,
#      by searching in a neighborhood (rectangle) of the ego agent.
#     :param scenario: Commonroad scenario
#     :param ego_state: State of ego
#     :param threshold_lateral: lateral distance to look for agents
#     :param threshold_longitudinal: longitudinal distance to look for agents
#     :return: List of interacting agents, as dynamic obstacles
#     """
#     interacting_dynamic_obstacles = []
#     # look for dynamic obstacles in radial direction
#     for dyn_obs in scenario.dynamic_obstacles:
#         dpos = ego_state.position - dyn_obs.initial_state.position
#         if np.linalg.norm(dpos) < threshold_radial:
#             interacting_dynamic_obstacles.append(dyn_obs.obstacle_id)
#     # increasing search radius until at least one interacting agent is found
#     # todo: replace this by game where Ego just choses its optimal solution ? (not joint?)
#     if not interacting_dynamic_obstacles:
#         interacting_dynamic_obstacles = interacting_lateral(scenario, ego_state, threshold_radial * 1.2)
#
#     # look for dynamic obstacles in longitudinal direction
#
#     return interacting_dynamic_obstacles


def convert_from_cr_state(state: State) -> VehicleState:
    return VehicleState(
        x=state.position[0],
        y=state.position[1],
        theta=state.orientation,
        vx=state.velocity,
        # delta=state.steering_angle  # just setting delta to 0, the cr State does not have 0 as data
        delta=0 #todo: fix this issue
    )


def convert_to_cr_state(vehicle_state: VehicleState) -> State:
    return State(
        position=np.array([vehicle_state.x, vehicle_state.y]),
        orientation=vehicle_state.theta,
        velocity=vehicle_state.vx,
        steering_angle=vehicle_state.delta,
    )


# todo 3: does this really get the correct state? i.e. the "initial one" according to current timestep? related to todo 1
def get_initial_states(ego_vehicle: EgoVehicle, scenario: Scenario, dyn_obs_list: List[DynamicObstacle]) \
        -> Mapping[PlayerName, VehicleState]:
    initial_states: Mapping[PlayerName, VehicleState] = {}
    dyn_obstacles = scenario.dynamic_obstacles
    initial_states[PlayerName("Ego")] = convert_from_cr_state(ego_vehicle.current_state)
    for dyn_obs in dyn_obs_list:
        initial_states[PlayerName(str(dyn_obs.obstacle_id))] = convert_from_cr_state(dyn_obs.initial_state)
    return initial_states


# basic version of reference generation. Not robust.
def generate_short_ref(state: State, network: LaneletNetwork):
    p = np.array((state.position[0], state.position[1]))
    return dglane_from_position(p=p, network=network)


def generate_route_ego(scenario: Scenario, planning_problem: PlanningProblem, plot_route: bool = False):
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    # plan routes, save multiple routes as list in candidate holder
    candidate_holder = route_planner.plan_routes()

    # here we retrieve the first route
    # option 1: retrieve first route
    route = candidate_holder.retrieve_first_route()
    # option 2: retrieve all routes
    # list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
    # option 3: retrieve the best route by orientation metric
    # route = candidate_holder.retrieve_best_route_by_orientation()

    # retrieve reference path from route
    ref_path = route.reference_path
    matplotlib.use("TkAgg")
    if plot_route:
        visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)
    # return ref_path
    route_ids = route.list_ids_lanelets
    assert len(route_ids) > 0, "list of reference lanes for Ego is empty!"
    ref_lanelet = scenario.lanelet_network.find_lanelet_by_id(route_ids[0])
    if len(route_ids) > 1:
        for lanelet_id in route_ids[1:]:
            new_lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            ref_lanelet = Lanelet.merge_lanelets(ref_lanelet, new_lanelet)

    return DgLanelet.from_commonroad_lanelet(ref_lanelet)


def generate_basic_refs(ego_vehicle: EgoVehicle, scenario: Scenario, planning_problem: PlanningProblem,
                        interacting_agents: List[DynamicObstacle]) \
        -> Mapping[PlayerName, RefLaneGoal]:
    ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}
    ego_init_state = ego_vehicle.current_state  # todo 4: when is current state different from init state? (related to 1&3)
    network = scenario.lanelet_network
    # ref_lanes[PlayerName("Ego")] = RefLaneGoal(generate_short_ref(ego_init_state, network), 1000.0)
    ego_ref_dglanelet = generate_route_ego(scenario, planning_problem)
    # todo: need to adapt metrics to use this
    ref_lanes[PlayerName("Ego")] = RefLaneGoal(ego_ref_dglanelet, 1000.0)
    ref_lanes[PlayerName("Ego")] = RefLaneGoal(ego_ref_dglanelet, ego_ref_dglanelet.get_lane_length() * goal_frac)

    for dyn_obs in interacting_agents:
        ref_lane = generate_short_ref(dyn_obs.initial_state, network)
        # todo: need to adapt metrics to use this
        # ref_lanes[PlayerName(str(agent_id))] = RefLaneGoal(ref_lane, 1000.0)
        ref_lanes[PlayerName(str(dyn_obs.obstacle_id))] = RefLaneGoal(ref_lane, ref_lane.get_lane_length() * goal_frac)
    return ref_lanes


def get_default_pref_structures(interacting_agents: List[DynamicObstacle]):
    pref_structures: Mapping[PlayerName, str] = {}
    default_str = "default_commonroad"
    # pref_structures[PlayerName("Ego")] = "default_commonroad_ego"
    pref_structures[PlayerName("Ego")] = "only_driving_area_violation"
    for dyn_obs in interacting_agents:
        pref_structures[PlayerName(str(dyn_obs.obstacle_id))] = default_str
    return pref_structures


def traj_gen_params_from_cr(cr_vehicle_params) -> TrajectoryGenParams:
    vp = VehicleParameters(
        vx_limits=(cr_vehicle_params.longitudinal.v_min, cr_vehicle_params.longitudinal.v_max),
        acc_limits=(-cr_vehicle_params.longitudinal.a_max, cr_vehicle_params.longitudinal.a_max),
        # todo 5: Correct? No max braking is given
        delta_max=cr_vehicle_params.steering.max,
        ddelta_max=cr_vehicle_params.steering.v_max
    )

    u_acc = frozenset([0.5])
    u_dst = frozenset([-0.3, 0.3])

    vg = VehicleGeometry(
        vehicle_type=CAR,
        m=1500.0,
        Iz=1300,
        w_half=cr_vehicle_params.w / 2.0,
        lf=cr_vehicle_params.l / 2.0,
        lr=cr_vehicle_params.l / 2.0,
        c_drag=0.3756,
        a_drag=2,
        e=0.5,
        color="royalblue",
    )

    params = TrajectoryGenParams(
        solve=False,
        s_final=goal_frac,  # todo: adapt metrics to use this
        # s_final=-1,
        max_gen=100,
        dt=D("0.2"),
        # keep at max 1 sec, increase k_maxgen in trajectory_generator for having more generations
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=vp.vx_limits[1],
        v_min=vp.vx_limits[0],
        st_max=vp.delta_max,
        dst_max=vp.ddelta_max,
        dt_samp=D("0.2"),
        dst_scale=False,
        n_factor=0.8,
        vg=vg,
    )

    return params


# todo 7: import ego_model and Vehicle Geometry from Commonroad
def get_traj_gen_params(interacting_agents: List[DynamicObstacle]):
    # read vehicle data from commonroad
    vehicle_type = VehicleType.FORD_ESCORT
    ego_params = vehicle_parameters[vehicle_type]
    # create trajectory generator parameters for all players involved
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {}
    traj_gen_params[PlayerName("Ego")] = traj_gen_params_from_cr(ego_params)
    for dyn_obs in interacting_agents:
        # todo 8: assuming all vehicles are Ford Escort.
        traj_gen_params[PlayerName(str(dyn_obs.obstacle_id))] = traj_gen_params_from_cr(ego_params)
    return traj_gen_params

########################IMPLEMENT LATER##############################
# def successor_lanelets(scenario: Scenario, position: np.ndarray):
#     lanelet_network = scenario.lanelet_network
#     current_lanelet_id = lanelet_network.find_lanelet_by_position([position])
#     current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet_id)
#     lanelets_succ, _ = current_lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet, lanelet_network)
#     return list(lanelets_succ)[0] # todo: check format
#
# def generate_refs(scenario: Scenario, interacting_lateral: List[int]) -> Mapping[PlayerName, RefLaneGoal]:
#     other_ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}
#     dyn_obstacles = scenario.dynamic_obstacles
#     for agent_id in interacting_lateral:
#         for dyn_obs in dyn_obstacles:
#             if dyn_obs.obstacle_id == agent_id:
#                 ref_lane = successor_lanelets(scenario, dyn_obs.initial_state.position)
#                 ref_dg_lane = DgLanelet(ref_lane)
#                 other_ref_lanes[agent_id] = RefLaneGoal(ref_dg_lane, 1.0)
#     return other_ref_lanes
#
# def get_ref_lanes(scenario: Scenario, interacting_lateral: List[int]) -> Mapping[PlayerName, RefLaneGoal]:
#     ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}
#     other_ref_lanes = generate_route_ego(scenario, interacting_lateral)
#     other_ref_lanes = generate_refs(scenario, interacting_lateral)
#     return ref_lanes
########################IMPLEMENT LATER##############################
