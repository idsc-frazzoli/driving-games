import math
import os
from decimal import Decimal as D
from typing import List, Mapping, Tuple

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

from dg_commons import PlayerName, SE2Transform
from dg_commons.maps import DgLanelet, LaneCtrPoint
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.model_structures import CAR
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from trajectory_games.structures import TrajectoryGenParams

goal_frac = 0.8


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


def interacting_agents(
        scenario: Scenario,
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
            obs_dist = np.linalg.norm(obs.initial_state.position - ego_state.position)  # todo: check this is correct
            if obs_dist < dist:
                closest_obs = obs
                dist = obs_dist
        leading_obs = [closest_obs]

        # make sure to count closest obstacle only once
        if leading_obs[0] in around_obs:
            around_obs.remove(leading_obs[0])

    obs_dict: Mapping[str, List[DynamicObstacle]] = {"leading": leading_obs, "around": around_obs}
    return obs_dict


# caution needed: since in simulation no delta is stored for vehicles, it is set to 0
def convert_from_cr_state(state: State) -> VehicleState:
    return VehicleState(
        x=state.position[0],
        y=state.position[1],
        theta=state.orientation,
        vx=state.velocity,
        # delta=state.steering_angle  # just setting delta to 0, the cr State does not have 0 as data
        delta=0
    )


def convert_to_cr_state(vehicle_state: VehicleState, time_step: int = 0) -> State:
    return State(
        position=np.array([vehicle_state.x, vehicle_state.y]),
        orientation=vehicle_state.theta,
        velocity=vehicle_state.vx,
        steering_angle=vehicle_state.delta,
        time_step=time_step,
    )


def get_initial_states(
        ego_vehicle: EgoVehicle,
        dyn_obs_list: List[DynamicObstacle]) -> Mapping[PlayerName, VehicleState]:
    initial_states: Mapping[PlayerName, VehicleState] = {
        PlayerName("Ego"): convert_from_cr_state(ego_vehicle.current_state)
    }

    for dyn_obs in dyn_obs_list:
        initial_states[PlayerName(str(dyn_obs.obstacle_id))] = convert_from_cr_state(dyn_obs.initial_state)

    return initial_states


def generate_refs_by_merging_successors(state: State, network: LaneletNetwork):
    p = np.array((state.position[0], state.position[1]))
    return dglane_from_position(p=p, network=network)


def generate_route_ego(scenario: Scenario, planning_problem: PlanningProblem, plot_route: bool = False):
    # careful: allow_diagonal is not tested according to Commonroad Library
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.PRIORITY_QUEUE, allow_diagonal=True)
    # plan routes, save multiple routes as list in candidate holder
    candidate_holder = route_planner.plan_routes()

    # here we retrieve the first route
    # option 1: retrieve first route
    route = candidate_holder.retrieve_first_route()
    # option 2: retrieve all routes
    # list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
    # option 3: retrieve the best route by orientation metric
    # route = candidate_holder.retrieve_best_route_by_orientation()

    matplotlib.use("TkAgg")
    # plot_route = True
    if plot_route:
        visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)

    def get_radius_from_lanelet(pos: np.ndarray) -> float:
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([pos])
        lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0][0])
        center_vertices = lanelet.center_vertices
        dist = 1000.0
        dist_idx = -100
        for i, vert in enumerate(center_vertices):
            ndist = np.linalg.norm(vert - pos)
            if ndist < dist:
                dist_idx = i
        r = np.linalg.norm(lanelet.left_vertices[dist_idx] - lanelet.right_vertices[dist_idx]) / 2.0
        return r


    n_ctrl_points = 50
    # remove first point and last 3 points in order to avoid going out of scenario boundaries
    idx = np.round(np.linspace(1, len(route.reference_path) - 3, n_ctrl_points)).astype(int)

    ps = route.reference_path[idx]
    theta = route.path_orientation[idx]
    rs = [get_radius_from_lanelet(p) for p in ps]

    qs = [SE2Transform(p=p, theta=theta) for p, theta in zip(ps, theta)]

    ctr_points = [LaneCtrPoint(q=q, r=r) for q, r in zip(qs, rs)]

    return DgLanelet(control_points=ctr_points), route.path_curvature


    # has a bug for certain routes, where routeplanner returns adjacent lanes (and not only successors)
    # # retrieve reference path from route
    # route_ids = route.list_ids_lanelets
    # assert len(route_ids) > 0, "list of reference lanes for Ego is empty!"
    # ref_lanelet = scenario.lanelet_network.find_lanelet_by_id(route_ids[0])
    # if len(route_ids) > 1:
    #     for lanelet_id in route_ids[1:]:
    #         new_lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
    #         ref_lanelet = Lanelet.merge_lanelets(ref_lanelet, new_lanelet)
    #
    # return DgLanelet.from_commonroad_lanelet(ref_lanelet)


def generate_ref_lanes(
        scenario: Scenario,
        planning_problem: PlanningProblem,
        inter_agents: List[DynamicObstacle]) -> Tuple[Mapping[PlayerName, RefLaneGoal], List[float]]:
    ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}

    network = scenario.lanelet_network

    ego_ref_dglanelet, route_curvature_ego = generate_route_ego(scenario, planning_problem)

    # todo: need to adapt metrics to use this correctly
    ref_lanes[PlayerName("Ego")] = RefLaneGoal(ego_ref_dglanelet, ego_ref_dglanelet.get_lane_length() * goal_frac)

    for dyn_obs in inter_agents:
        ref_lane = generate_refs_by_merging_successors(dyn_obs.initial_state, network)
        ref_lanes[PlayerName(str(dyn_obs.obstacle_id))] = RefLaneGoal(ref_lane, ref_lane.get_lane_length())
    return ref_lanes, route_curvature_ego


def get_default_pref_structures(inter_agents: List[DynamicObstacle]):
    pref_structures: Mapping[PlayerName, str] = {}
    default_str = "default_commonroad"

    pref_structures[PlayerName("Ego")] = "default_commonroad_ego"

    for dyn_obs in inter_agents:
        pref_structures[PlayerName(str(dyn_obs.obstacle_id))] = default_str
    return pref_structures


def traj_gen_params_from_cr(cr_vehicle_params, is_ego: bool) -> TrajectoryGenParams:
    vp = VehicleParameters(
        vx_limits=(0.0, cr_vehicle_params.longitudinal.v_max),  # don't allow backwards driving
        acc_limits=(-cr_vehicle_params.longitudinal.a_max, cr_vehicle_params.longitudinal.a_max),
        # todo 5: Correct? No max braking is given
        delta_max=cr_vehicle_params.steering.max,
        ddelta_max=cr_vehicle_params.steering.v_max
    )

    v_switch = cr_vehicle_params.longitudinal.v_switch

    if is_ego:
        u_acc = frozenset([-3.0])
        u_dst = frozenset([0.0])

    else:
        u_acc = frozenset([-1.0])
        u_dst = frozenset([0.0])

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
        s_final=-1,  # todo: adapt metrics to use this
        max_gen=5,
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
        v_switch=v_switch
    )

    return params


# todo 7: import ego_model and Vehicle Geometry from Commonroad
def get_traj_gen_params(interacting_agents: List[DynamicObstacle]):
    # read vehicle data from commonroad
    vehicle_type = VehicleType.FORD_ESCORT
    ego_params = vehicle_parameters[vehicle_type]
    # create trajectory generator parameters for all players involved
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {}
    traj_gen_params[PlayerName("Ego")] = traj_gen_params_from_cr(ego_params, is_ego=True)
    for dyn_obs in interacting_agents:
        # todo 8: assuming all vehicles are Ford Escort.
        traj_gen_params[PlayerName(str(dyn_obs.obstacle_id))] = traj_gen_params_from_cr(ego_params, is_ego=False)
    return traj_gen_params
