from typing import List, Tuple, Optional
import math
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from duckietown_world import SE2Transform
from zuper_commons.types import ZException

from dg_commons.planning.lanes import DgLanelet, LaneCtrPoint
from sim.agents.agent import Agent
from sim.agents.lane_follower import LFAgent
from sim.models import Pacejka
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn, VehicleParametersDyn
from sim.models.vehicle_structures import VehicleGeometry, CAR
import numpy as np


class NotSupportedConversion(ZException):
    pass


def model_agent_from_dynamic_obstacle(dyn_obs: DynamicObstacle, lanelet_network: LaneletNetwork) -> (
        VehicleModelDyn, Agent):
    """
    This function aims to create a non-playing character (fixed sequence of commands) in our simulation environment from
    a dynamic obstacle of commonroad (fixed sequence of states).
    # fixme currently only cars are supported
    # fixme this function needs to be improved...
    :param dyn_obs:
    :param lanelet_network:
    :return:
    """
    if not dyn_obs.obstacle_type == ObstacleType.CAR:
        raise NotSupportedConversion(commonroad=dyn_obs.obstacle_type)

    axle_length_ratio = .8  # the distance between wheels is less than the car body
    l = dyn_obs.obstacle_shape.length * axle_length_ratio
    dtheta = dyn_obs.prediction.trajectory.state_list[0].orientation - dyn_obs.initial_state.orientation
    # todo some scaling
    delta = dtheta / l
    x0 = VehicleStateDyn(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                         theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                         delta=delta)
    mass, rot_inertia = _estimate_mass_inertia(length=dyn_obs.obstacle_shape.length, width=dyn_obs.obstacle_shape.width)
    axle_width_ratio = .95  # the distance between wheels is less than the car body
    w_half = dyn_obs.obstacle_shape.width / 2 * axle_width_ratio
    vg = VehicleGeometry(vehicle_type=CAR, w_half=w_half, m=mass, Iz=rot_inertia, lf=l / 2.0,
                         lr=l / 2.0, e=0.6, c_drag=0.3756, c_rr_f=0.003, c_rr_r=0.003, a_drag=2, color="royalblue")
    vp = VehicleParametersDyn.default_car()
    model = VehicleModelDyn(x0=x0,
                            vg=vg,
                            vp=vp,
                            pacejka_front=Pacejka.default_car_front(),
                            pacejka_rear=Pacejka.default_car_rear())

    # Agent
    dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, lanelet_net=lanelet_network)
    agent = LFAgent(dglane)
    return model, agent


def _estimate_mass_inertia(length: float, width: float) -> Tuple[float, float]:
    """ #todo justify and fix this empirical formulas """
    alpha = 50
    beta = 1.6
    area = length * width
    mass = alpha * area ** beta
    inertia = mass * (length + width) / 6
    return mass, inertia


def infer_lane_from_dyn_obs(dyn_obs: DynamicObstacle, lanelet_net: LaneletNetwork) -> DgLanelet:
    """ Tries to find a lane corresponding to the trajectory, if no lane is found it creates one from the trajectory"""

    initial_state = dyn_obs.initial_state.position
    end_state = dyn_obs.prediction.trajectory.state_list[-1].position

    initial_ids = lanelet_net.find_lanelet_by_position([initial_state])[0]
    final_ids = lanelet_net.find_lanelet_by_position([end_state])[0]
    return_lanelet: Optional[DgLanelet] = None

    for initial_id in initial_ids:
        lanelet = lanelet_net.find_lanelet_by_id(initial_id)
        merged_lanelets, merged_ids = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanelet_net)
        for i, merged_id in enumerate(merged_ids):
            if any(item in merged_id for item in final_ids):
                return_lanelet = DgLanelet.from_commonroad_lanelet(merged_lanelets[i])
                break
        if return_lanelet is not None:
            break

    if return_lanelet is None:
        return_lanelet = dglanelet_from_trajectory(dyn_obs.prediction.trajectory.state_list, lanelet_net)

    return return_lanelet


def dglanelet_from_trajectory(states, lanelet_net: LaneletNetwork, width: Optional[float] = None):
    if width is None:
        width = compute_width(lanelet_net)

    control_points: List[LaneCtrPoint] = []
    for state in states:
        q = SE2Transform(p=state.position, theta=state.orientation)
        control_points.append(LaneCtrPoint(
            q=q, r=width / 2))

    add_control_points = additional_trajectory_steps(control_points[-1].q, lanelet_net)

    control_points = control_points + add_control_points

    return DgLanelet(control_points=control_points)


def compute_width(lanelet_net: LaneletNetwork):
    res = 0
    counter = 0
    for lanelet in lanelet_net.lanelets:
        weight, _ = lanelet.right_vertices.shape
        res += np.average(np.linalg.norm(lanelet.right_vertices - lanelet.left_vertices, axis=1)) * weight
        counter += weight
    return res/counter


def additional_trajectory_steps(end_pose: SE2Transform, lanelet_net: LaneletNetwork):
    end_position = end_pose.p
    current_lanelet_id = lanelet_net.find_lanelet_by_position([end_position])[0][0]
    current_lanelet = lanelet_net.find_lanelet_by_id(current_lanelet_id)
    merged_lanelets, _ = current_lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet,
                                                                                         lanelet_net)
    merged_lanelet: Lanelet = merged_lanelets[0]
    dg_lanelet = DgLanelet.from_commonroad_lanelet(merged_lanelet)
    beta, _ = dg_lanelet.find_along_lane_closest_point(end_position)
    control_points = dg_lanelet.control_points[math.ceil(beta):]

    return control_points

