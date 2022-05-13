import math
from dataclasses import dataclass
from itertools import combinations
from typing import Tuple, List, Dict, Callable, Set, Mapping

import commonroad_dc.pycrcc as pycrcc
import geometry as geo
import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficLight, TrafficLightState
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.scenario.trajectory import State
from commonroad_dc.boundary import boundary
from commonroad_dc.collision.trajectory_queries import trajectory_queries
# todo: fix packages s.t. this import is possible
# from commonroad_challenge.utils import interacting_agents, convert_to_cr_state
from frozendict import frozendict
from shapely.geometry import Point, LineString, Polygon

from dg_commons import PlayerName, SE2Transform, seq_differentiate, apply_SE2_to_shapely_geo, Timestamp
from dg_commons.maps import DgLanePose, DgLanelet
from dg_commons.planning import Trajectory, JointTrajectories, RefLaneGoal
from dg_commons.seq.sequence import DgSampledSequence
from dg_commons.sim.models.vehicle import VehicleGeometry, VehicleState
from dg_commons.time import time_function
from driving_games.metrics_structures import (
    EvaluatedMetric,
    JointEvaluatedMetric,
    Metric,
    MetricEvaluationContext,
    PlayerEvaluatedMetrics,
    JointPlayerEvaluatedMetrics,
)
from .trajectory_world import TrajectoryWorld

__all__ = [
    "get_metrics_set",
    "EpisodeTime",
    "DeviationLateral",
    "DeviationHeading",
    "DrivableAreaViolation",
    "ProgressAlongReference",
    "LongitudinalAcceleration",
    "LateralComfort",
    "SteeringAngle",
    "SteeringRate",
    "Clearance",
    "CollisionEnergy",
    "ClearanceViolationTime",
    "MinimumClearance",
    "MetricEvaluation",
    "GoalViolation",
    "DistanceToObstacle_CR",
    "SteeringRateSquared",
    "DeviationLateralSquared",
    "LongitudinalAccelerationSquared",
    "RoadCompliance_CR",
    "SafeDistance_CR",
    "MaximumVelocity_CR"
]


# todo: remove when import above is fixed
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


# todo: remove when import above is fixed
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


# todo: remove when import above is fixed
def convert_to_cr_state(vehicle_state: VehicleState, time_step: int = 0) -> State:
    return State(
        position=np.array([vehicle_state.x, vehicle_state.y]),
        orientation=vehicle_state.theta,
        velocity=vehicle_state.vx,
        steering_angle=vehicle_state.delta,
        time_step=time_step,
    )


# todo: remove when import above is fixed
def interacting_agents(
        scenario: Scenario,
        ego_state: State,
        look_ahead_dist: float,
        around_dist_r: float,
        around_dist_f: float,
        around_dist_lat: float,
        only_leading=False) -> Mapping[str, List[DynamicObstacle]]:
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
    around_obs = []

    if len(leading_obs) > 1:
        # only keep closest leading obstacle
        dist = 99999.
        closest_obs = None
        for obs in leading_obs:
            obs_dist = np.linalg.norm(obs.initial_state.position - ego_state.position)
            if obs_dist < dist:
                closest_obs = obs
                dist = obs_dist
        leading_obs = [closest_obs]

        if not only_leading:
            # find dyn. obstacles interacting with ego by being close to it
            around_area_of_interest = rectangle_around_ego(ego_state=ego_state,
                                                           look_forward_dist=around_dist_f,
                                                           look_backward_dist=around_dist_r,
                                                           look_lateral_dist=around_dist_lat)

            around_obs = filter_obstacles(scenario, around_area_of_interest)

            # make sure to count closest obstacle only once
            if leading_obs[0] in around_obs:
                around_obs.remove(leading_obs[0])

    obs_dict: Mapping[str, List[DynamicObstacle]] = {"leading": leading_obs, "around": around_obs}
    return obs_dict


def get_evaluated_metric(players: List[PlayerName], f: Callable[[PlayerName], EvaluatedMetric]) -> JointEvaluatedMetric:
    mer: Dict[PlayerName, EvaluatedMetric] = {}
    for player_name in players:
        mer[player_name] = f(player_name)
    return mer


class EpisodeTime(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "Length of the episode (smaller preferred)."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            ret = self.get_metric(seq=DgSampledSequence[float](values=traj.timestamps, timestamps=traj.timestamps))
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationLateral(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the lateral deviation from reference path."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_sn = context.points_curv[player]
            abs_n = [_.distance_from_center for _ in traj_sn]
            ret = self.get_integrated_metric(seq=DgSampledSequence[float](
                values=abs_n, timestamps=traj.get_sampling_points()))
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationLateralSquared(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the squared lateral deviation from reference path."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_sn = context.points_curv[player]
            abs_n = [_.distance_from_center ** 2 for _ in traj_sn]
            ret = self.get_integrated_metric(seq=DgSampledSequence[float](
                values=abs_n, timestamps=traj.get_sampling_points()))
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationHeading(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the heading deviation from reference path."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            timestamps = traj.get_sampling_points()
            traj_sn = context.points_curv[player]
            head = [abs(_.relative_heading) for _ in traj_sn]
            ret = self.get_integrated_metric(seq=DgSampledSequence[float](timestamps=timestamps, values=head))
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DrivableAreaViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the drivable area violation by the vehicle."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_sn = context.points_curv[player]
            player_geo = context.geos[player]

            def get_violation(curv: DgLanePose, geom: VehicleGeometry) -> float:
                # checking left boundary
                if curv.lateral > 0:
                    violation = curv.lateral + geom.w_half - curv.lateral_left
                    if violation > 0:
                        return violation
                # checking right boundary
                if curv.lateral < 0:
                    violation = curv.lateral_right - curv.lateral + geom.w_half
                    if violation > 0:
                        return violation

                return 0.0

            viol = [get_violation(curv=x, geom=player_geo) for x in traj_sn]
            viol_seq = DgSampledSequence[float](timestamps=traj.get_sampling_points(), values=viol)
            ret = self.get_integrated_metric(viol_seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ProgressAlongReference(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = (
        "This metric computes how far the robot drove **along the reference path** (negative for smaller preferred)"
    )

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_sn = context.points_curv[player]
            # negative for smaller preferred
            final_progress = [traj_sn[0].along_lane - traj_sn[-1].along_lane]
            ret = EvaluatedMetric(
                name=self.get_name(),
                value=final_progress[0]
            )
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAcceleration(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the longitudinal acceleration the robot."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_vel = traj.transform_values(lambda x: x.vx, float)
            acc_seq = seq_differentiate(traj_vel)
            ret = self.get_integrated_metric(acc_seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAccelerationSquared(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the squared longitudinal acceleration of the robot."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            traj_vel = traj.transform_values(lambda x: x.vx * x.vx, float)
            acc_seq = seq_differentiate(traj_vel)
            ret = self.get_integrated_metric(acc_seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


def _get_lat_comf(x: VehicleState) -> float:
    return abs(x.vx * x.delta)


class LateralComfort(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the lateral comfort of the robot."
    scale: float = 0.5

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            lat_comf_seq = traj.transform_values(_get_lat_comf, float)
            ret = self.get_integrated_metric(seq=lat_comf_seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringAngle(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the steering angle of the robot."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            st_seq = traj.transform_values(lambda x: abs(x.delta), float)
            ret = self.get_integrated_metric(seq=st_seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRate(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the rate of change of steering angle of the robot."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            st_traj = traj.transform_values(lambda x: x.delta, float)
            dst = seq_differentiate(st_traj)
            ret = self.get_integrated_metric(dst)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRateSquared(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the squared rate of change of the steering angle of the robot."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            st_traj = traj.transform_values(lambda x: x.delta * x.delta, float)
            dst = seq_differentiate(st_traj)
            ret = self.get_integrated_metric(dst)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class GoalViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metrics computes how much the observed trajectory violates the preferred reference" \
                  " lane in favour of another lane."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]
            # check there is more than one reference lane
            goals = context.goals[player]
            points_cart = context.points_cart[player]
            integrated_abs_n: Mapping[RefLaneGoal, EvaluatedMetric] = {}
            if len(context.goals[player]) > 1:
                for goal in goals:
                    # this metric only works for RefLaneGoals for now
                    assert isinstance(goal, RefLaneGoal), "Only RefLaneGoal(s) can be used for this metric"
                    points_curv: List[DgLanePose] = [goal.ref_lane.lane_pose_from_SE2Transform(q) for q in points_cart]
                    abs_n = [_.distance_from_center for _ in points_curv]
                    integrated_abs_n[goal] = self.get_integrated_metric(seq=DgSampledSequence[float](
                        values=abs_n, timestamps=traj.get_sampling_points()))

                # find smallest cumulative distance
                min_value = min([val.value for val in integrated_abs_n.values()])
                if integrated_abs_n[goals[0]].value == min_value:
                    return EvaluatedMetric(name=self.get_name(), value=0)
                else:
                    return EvaluatedMetric(name=self.get_name(), value=min_value)
            else:
                return EvaluatedMetric(name=self.get_name(), value=0)

        return get_evaluated_metric(context.get_players(), calculate_metric)


def crossing_line_from_pos(pos: np.ndarray, lanelet: DgLanelet) -> Tuple[float, float]:
    """Takes a point in a lanelet, computes the closest points on each side and then a line connecting those point
        Returns coefficients (a,b) of y=ax+b"""


class TrafficLightViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric evaluates if a red traffic light has been passed. Penalty is proportional to speed." \
                  "A check for existence of traffic lights is executed on the lanelet corresponding to the initial" \
                  "position of the trajectory."

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            def _is_light_red(time_step: int, light: TrafficLight) -> bool:
                state = light.get_state_at_time_step(time_step=time_step)
                return state == TrafficLightState.RED

            traj: Trajectory = context.trajectories[player]
            if traj in self.cache:
                return self.cache[traj]

            network = context.dgscenario.lanelet_network
            initial_pos = np.array([traj.values[0].x, traj.values[0].y])

            initial_lanelet_id = network.find_lanelet_by_position(initial_pos)
            # todo[LEON]: figure out what happens (error?) if it does not intersect, continue until one is intersecting
            initial_lanelet = network.find_lanelet_by_id(initial_lanelet_id)

            traffic_lights = initial_lanelet.traffic_lights
            zero_violation = EvaluatedMetric(name=self.get_name(), value=0)
            # if there are no traffic lights, there is no violation:
            if traffic_lights == set():
                self.cache[traj] = zero_violation
                return zero_violation
            # if there are lights, evaluate if there is a violation
            else:
                traffic_lights = list(traffic_lights)
                dg_lanelet = DgLanelet.from_commonroad_lanelet(initial_lanelet)
                traffic_lights_se2 = [SE2Transform(p=traffic_light.position, theta=0.0) for traffic_light in
                                      traffic_lights]
                lights_progress = [dg_lanelet.lane_pose_from_SE2Transform(se2) for se2 in traffic_lights_se2]

                # now check if trajectory is surpassing any traffic lights
                points_curv = context.points_curv[player]
                for i, (point1, point2) in enumerate(zip(points_curv[:-1], points_curv[1:])):
                    for j, light_pose in enumerate(lights_progress):
                        if point1.along_lane < light_pose.along_lane < point2.along_lane:
                            violation = _is_light_red(time_step=i, light=traffic_lights[j])
                            if violation:
                                values = list(traj.values)
                                avg_speed = abs(values[i + 1].vx - values[i].vx) / 2
                                ret = EvaluatedMetric(name=self.get_name(), value=avg_speed)
                                self.cache[traj] = ret
                                return ret

            # if nothing was returned above, no violation detected
            self.cache[traj] = zero_violation
            return zero_violation

        return get_evaluated_metric(context.get_players(), calculate_metric)


PairwiseValues = Dict[Tuple[PlayerName, PlayerName], DgSampledSequence]


class Clearance(Metric):
    description = "This metric computes the clearance between players (pairwise). If two trajectories have" \
                  " different lengths, the clearance is only computed until the end timeof the shortest trajectory."

    sampling_time: Timestamp = 1.0
    clearance_tolerance: float = 20.0  # if distance between CoM of two vehicles is greater, approximate clearance

    @staticmethod
    def get_clearance(clearance_tolerance: float, states: Tuple[VehicleState, VehicleState],
                      geos: Tuple[VehicleGeometry, VehicleGeometry]):

        q1 = SE2Transform(p=[states[0].x, states[0].y], theta=states[0].theta)
        q2 = SE2Transform(p=[states[1].x, states[1].y], theta=states[1].theta)

        # if clearance is large, return approximated value to save computational resources
        if np.linalg.norm(q1.p - q2.p) > clearance_tolerance:
            return np.linalg.norm(q1.p - q2.p)
        else:
            a_shape = geos[0].outline_as_polygon
            b_shape = geos[1].outline_as_polygon
            a_shape_tra = apply_SE2_to_shapely_geo(a_shape, q1.as_SE2())
            b_shape_tra = apply_SE2_to_shapely_geo(b_shape, q2.as_SE2())
            return a_shape_tra.distance(b_shape_tra)

    def calculate_all_clearances(self, context: MetricEvaluationContext) -> PairwiseValues:
        joint_values: PairwiseValues = {}

        for pair in combinations(context.get_players(), r=2):
            t_start = float(context.trajectories[pair[0]].get_start())
            assert t_start == context.trajectories[pair[1]].get_start(), "Trajectories need to start at the same time."
            #
            # assert t_start == context.trajectories[pair[1]].get_start() and \
            #        t_end == context.trajectories[pair[1]].get_end(), \
            #     "The start and end time of different trajectories needs to be the same"

            t_end_0 = float(context.trajectories[pair[0]].get_end())
            t_end_1 = float(context.trajectories[pair[1]].get_end())

            t_end = min([t_end_0, t_end_1])

            clearance = []
            n_points = int((t_end - t_start) / self.sampling_time)
            # up-sample going backwards and reverse
            timestamps = [t_end - i * self.sampling_time for i in range(n_points + 1)]
            timestamps.reverse()

            geos = (context.geos[pair[0]], context.geos[pair[1]])

            for t in timestamps:
                states = (context.trajectories[pair[0]].at_interp(t), context.trajectories[pair[1]].at_interp(t))
                clearance.append(
                    self.get_clearance(clearance_tolerance=self.clearance_tolerance, states=states, geos=geos))

            joint_values[pair[0], pair[1]] = DgSampledSequence(values=clearance, timestamps=timestamps)

        return joint_values

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        clearances = self.calculate_all_clearances(context=context)
        timestamps = list(clearances.values())[0].timestamps

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            clearance = np.array([0 for _ in range(len(timestamps))])
            for player_pair in clearances.keys():
                if player in player_pair:
                    clear = np.array(clearances[player_pair].values)
                    clearance = np.add(clearance, clear)
            seq = DgSampledSequence[float](values=clearance, timestamps=timestamps)
            ret = self.get_integrated_metric(seq=seq)
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


def cr_dist(clearance: float, w_dist: float = 0.2):
    return np.exp(clearance * w_dist)


class DistanceToObstacle_CR(Clearance):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the distance to other obstacles as specified in J_D, CommonRoad costs"
    sampling_time: Timestamp = 0.1
    clearance_tolerance: float = 10.0  # if distance between CoM of two vehicles is greater, approximate clearance

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:

        clearances = self.calculate_all_clearances(context=context)
        if not clearances == {}:
            timestamps = list(clearances.values())[0].timestamps

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            clearance_seq = []

            if clearances == {}:
                return EvaluatedMetric(name=self.get_name(), value=0)
            for t in timestamps:
                min_clear = 10000.0
                for player_pair in clearances.keys():
                    if player in player_pair and clearances[player_pair].at_interp(t) < min_clear:
                        min_clear = clearances[player_pair].at_interp(t)
                assert min_clear < 10000.0, "At least one clearance must be smaller than this."
                clearance_seq.append(cr_dist(min_clear))

            seq = DgSampledSequence[float](values=clearance_seq, timestamps=timestamps)
            ret = self.get_integrated_metric(seq=seq)

            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class MinimumClearance(Clearance):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric returns a cost (inverse to clearance)" \
                  " when minimum clearance is not respected between agents."
    sampling_time: Timestamp = 1.0
    clearance_tolerance: float = 20.0  # if distance between CoM of two vehicles is greater, approximate clearance
    min_clearance: float = 10.0

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:

        clearances = self.calculate_all_clearances(context=context)
        timestamps = list(clearances.values())[0].timestamps

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            clearance_seq = np.array([0 for _ in range(len(timestamps))])
            for player_pair in clearances.keys():
                if player in player_pair:
                    clear = np.array(clearances[player_pair].values)
                    idx_good = clear > self.min_clearance
                    clear[idx_good] = 0.0
                    clear[~idx_good] = self.min_clearance * np.ones(np.shape(clear[~idx_good])) - clear[~idx_good]
                    clearance_seq = np.add(clearance_seq, clear)
            seq = DgSampledSequence[float](values=clearance_seq, timestamps=timestamps)
            ret = self.get_integrated_metric(seq=seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ClearanceViolationTime(Clearance):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the time a minimum clearance is violated between agents."
    sampling_time: Timestamp = 1.0
    clearance_tolerance: float = 20.0  # if distance between CoM of two vehicles is greater, approximate clearance
    min_clearance: float = 10.0

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:

        clearances = self.calculate_all_clearances(context=context)
        timestamps = list(clearances.values())[0].timestamps

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            clearance_seq = np.array([0 for _ in range(len(timestamps))])
            for player_pair in clearances.keys():
                if player in player_pair:
                    clear = np.array(clearances[player_pair].values)
                    idx_bad = clear <= self.min_clearance
                    clear[idx_bad] = self.sampling_time  # approximation: could violate Bellman optimality
                    clear[~idx_bad] = 0.0
                    clearance_seq = np.add(clearance_seq, clear)
            seq = DgSampledSequence[float](values=clearance_seq, timestamps=timestamps)
            ret = self.get_integrated_metric(seq=seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


# from collisions_check.py
def get_2d_velocity(x: VehicleState) -> geo.T2value:
    """Transform longitudinal velocity norm into 2d vector"""
    v_l = np.array([x.vx, 0])
    rot: geo.SO2value = geo.SO2_from_angle(x.theta)
    v_g = rot @ v_l
    return v_g


Crashes = Mapping[Tuple[PlayerName, PlayerName], Timestamp]


# todo make better test since metric has changed
class CollisionEnergy(Clearance):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the energy of collision between agents."
    sampling_time: Timestamp = 0.2
    clearance_tolerance: float = 20.0  # if distance between CoM of two vehicles is greater, approximate clearance
    min_clearance: float = 10.0

    @staticmethod
    def get_collision_energy(geometries: Tuple[VehicleGeometry, VehicleGeometry],
                             states: Tuple[VehicleState, VehicleState]) -> float:

        vel_1, vel_2 = get_2d_velocity(states[0]), get_2d_velocity(states[1])
        energy = 0.5 * (geometries[0].m + geometries[1].m) * np.linalg.norm(vel_1 - vel_2) ** 2
        return energy

    def crashes_taking_place(self, context: MetricEvaluationContext, tol: float = 1e-3) -> Crashes:
        flag = 99999999

        # keep first time clearance is 0 (with tolerance) for each player pair
        def compute_first_collisions() -> Crashes:
            crashes: Crashes = {}
            for p_pair, pair_clearance in clearances.items():
                # account for numerical errors
                first_crash = np.where(np.array(pair_clearance.values) < tol)[0].tolist()
                if first_crash:
                    first_crash = first_crash[0]
                    crashes[p_pair] = first_crash

            return crashes

        clearances = self.calculate_all_clearances(context=context)
        hyp_crashes = compute_first_collisions()

        actual_crashes: Crashes = {}
        # find first crash for each player (discard pairs where the crash happens subsequently)
        for player in context.get_players():
            player_idx = []
            # compute the earliest collision (checking against collisions with all players)
            for player_pair, idx_crash in hyp_crashes.items():
                if player in player_pair:
                    player_idx.append(idx_crash)

            # if player is not affected by any collision, continue to next player
            if not player_idx:
                continue

            min_crash_idx = min(player_idx)

            for player_pair, idx_crash in hyp_crashes.items():
                if player in player_pair and idx_crash > min_crash_idx:
                    hyp_crashes[player_pair] = flag  # flag for removal

        # now only keep all crashes that were not filtered out previously
        for key, value in hyp_crashes.items():
            if value != flag:
                actual_crashes[key] = value

        return actual_crashes

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:

        clearances = self.calculate_all_clearances(context=context)  # todo calculated twice, make more efficient
        timestamps = list(clearances.values())[0].timestamps
        crashes = self.crashes_taking_place(context=context)

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            coll_energies_seq = np.array([0 for _ in range(len(timestamps))])
            for player_pair, crash_idx in crashes.items():
                if player in player_pair:
                    geos = (context.geos[player_pair[0]], context.geos[player_pair[1]])
                    state_1 = context.trajectories[player_pair[0]].values[crash_idx]
                    state_2 = context.trajectories[player_pair[1]].values[crash_idx]
                    crash_states = (state_1, state_2)
                    coll_energy = self.get_collision_energy(geometries=geos, states=crash_states)
                    coll_energies_seq[crash_idx] = + coll_energy

            seq = DgSampledSequence[float](values=coll_energies_seq, timestamps=timestamps)
            ret = self.get_integrated_metric(seq=seq)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class CollisionBool(CollisionEnergy):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes if a collision occurs or not."
    sampling_time: Timestamp = 0.2
    clearance_tolerance: float = 20.0  # if distance between CoM of two vehicles is greater, approximate clearance
    min_clearance: float = 10.0

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:

        # clearances = self.calculate_all_clearances(context=context)  # todo calculated twice, make more efficient
        crashes = self.crashes_taking_place(context=context)

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            collision = 0.0
            for player_pair, crash_idx in crashes.items():
                if player in player_pair:
                    collision = 1.0

            return EvaluatedMetric(name=self.get_name(), value=collision)

        return get_evaluated_metric(context.get_players(), calculate_metric)


# todo: write test
class AngularViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the deviation (in radians) from a circular sector"

    def __init__(self, min_angle: float, max_angle: float):
        self.min_angle = min_angle
        self.max_angle = max_angle

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            headings = [heading.theta for heading in traj.values]
            for idx, head in enumerate(headings):
                if self.min_angle < head < self.max_angle:
                    headings[idx] = 0
                elif head > self.max_angle:
                    headings[idx] = head - self.max_angle
                elif head < self.min_angle:
                    headings[idx] = - self.max_angle - head
            ret = self.get_integrated_metric(seq=DgSampledSequence[float](
                values=headings, timestamps=traj.get_sampling_points()))
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


@dataclass
class RSS1_Parameters:
    """
    All accelerations and braking values are positive.
    :param r_response_time: Response time of the rear vehicle
    :param max_acc_r: Maximum acceleration during the response time
    :param min_breaking_r: Minimum braking deceleration until full stop, after reaction time
    :param max_braking_f: max braking of vehicle in front
    """
    r_response_time: Timestamp
    max_acc_r: float
    min_breaking_r: float
    max_braking_f: float


class RSS1_CR(Metric):
    # cache: Dict[Trajectory, EvaluatedMetric] = {} # don't use cache, as same trajectory might have different
    # surrounding dynamic obstacles
    description = "This metric computes RSS1 (Safe distance) metric for commonroad scenarios."

    def __init__(self, params: RSS1_Parameters):
        self.params = params

    def RSS1(self, vr: float, vf: float, params: RSS1_Parameters) -> float:
        """
        First RSS Metric (Safe Distance). r: rear vehicle (ego), f: front vehicle (dynamic obstacle of Commonroad).
        All accelerations and braking values are positive.
        :param vr: Velocity of rear vehicle
        :param vf: Velocity of front vehicle
        :param params: parameters for computing RSS1
        :return: minimum safe distance
        """

        return vr * params.r_response_time \
               + 0.5 * params.max_acc_r * params.r_response_time ** 2 \
               + ((vr + params.r_response_time * params.max_acc_r) ** 2) / (2 * params.min_breaking_r) \
               - 1.0 * vf * vf / (2.0 * params.max_braking_f)

    def compute_RSS1_trajectory_dynobs(self, traj: Trajectory, dyn_obs_state: State, params: RSS1_Parameters):
        """
        Compute RSS1 between a trajectory for the trailing vehicle and a commonroad dynamic obstacle (leading vehicle).
        Constant velocity and orientation assumption is made for leading vehicle.

        :param traj: trajectory of trailing vehicle
        :param dyn_obs_state: state of dynamic obstacle
        :param params: parameters of RSS1 (reaction time, accelerations and braking)
        :return: 1 if distance between final step of trajectory is smaller than RSS1, 0 otherwise
        """
        last_traj_state = traj.values[-1]
        last_traj_position = np.array([last_traj_state.x, last_traj_state.y])
        dt = traj.timestamps[-1] - traj.timestamps[0]
        dyn_obs_init_pos = dyn_obs_state.position
        dyn_obs_final_pos_x = dyn_obs_init_pos[0] + math.cos(dyn_obs_state.orientation) * dyn_obs_state.velocity * dt
        dyn_obs_final_pos_y = dyn_obs_init_pos[1] + math.sin(dyn_obs_state.orientation) * dyn_obs_state.velocity * dt
        dyn_obs_final_pos = np.array([dyn_obs_final_pos_x, dyn_obs_final_pos_y])

        final_distance = np.linalg.norm(dyn_obs_final_pos - last_traj_position)

        rss_distance = self.RSS1(vr=last_traj_state.vx, vf=dyn_obs_state.velocity, params=params)
        return int(final_distance < rss_distance)

    def get_leading_obs(self, scenario: Scenario, state: VehicleState) -> DynamicObstacle:
        pass

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            scenario = context.dgscenario.scenario
            player_state = traj.values[0]
            leading_dyn_obs = self.get_leading_obs(scenario, player_state)  # returns obstacle in front

            viol = self.compute_RSS1_trajectory_dynobs(traj, leading_dyn_obs.initial_state, self.params)

            ret = EvaluatedMetric(
                name=self.get_name(),
                value=viol,
            )
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class RoadCompliance_CR(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metrics computes if a trajectory is compliant with the road, i.e. if the vehicle geometry" \
                  "is entirely withing the road. Using CR tools."
    road_boundary = None
    scenario_id = None

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        # reset road boundary when scenario changes
        if context.dgscenario.scenario.scenario_id != self.scenario_id:
            self.road_boundary = None
            self.scenario_id = context.dgscenario.scenario.scenario_id

        if self.road_boundary is None:
            road_boundary_obstacle, self.road_boundary = boundary.create_road_boundary_obstacle(
                context.dgscenario.scenario)

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]

            car_half_width = context.geos[player].width / 2.0
            car_half_length = (context.geos[player].lr + context.geos[player].lf) / 2.0

            vehicle_states = traj.values
            states = [[state.x, state.y, state.theta] for state in vehicle_states]

            # From CR: create time-varying obstacle for "player" car
            def create_tvobstacle(traj_list, car_half_length, car_half_width):
                tvo = pycrcc.TimeVariantCollisionObject(0)
                for traj in traj_list:
                    tvo.append_obstacle(pycrcc.RectOBB(car_half_length, car_half_width, traj[2], traj[0], traj[1]))
                return tvo

            co = create_tvobstacle(states, car_half_length, car_half_width)

            # From CR: preprocess using OBB sum hull
            preprocessed_trajectory, err = trajectory_queries.trajectory_preprocess_obb_sum(co)
            if (err):
                raise Exception("trajectory preprocessing error")

            # From CR: compute time step of collision with road boundary
            ret = trajectory_queries.trajectories_collision_static_obstacles([preprocessed_trajectory],
                                                                             self.road_boundary,
                                                                             method='grid',
                                                                             num_cells=32,
                                                                             auto_orientation=True)

            # if ret[0] == -1, no collision with the road boundary has occurred
            if ret[0] == -1:
                ret = 0.0
            # if ret[0] >= 0, collision with the road boundary has occurred at time_step = ret[0]
            else:
                ret = 1.0

            return EvaluatedMetric(name=self.get_name(), value=ret)

        return get_evaluated_metric(context.get_players(), calculate_metric)


class MaximumVelocity_CR(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metrics computes the violation of the maximum allowed speed, given as attribute of a " \
                  "commonroad lanelet."
    maximum_velocities = None
    scenario_id = None

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        # recompute velocities when scenario changes
        if context.dgscenario.scenario.scenario_id != self.scenario_id:
            self.maximum_velocities = None
            self.scenario_id = context.dgscenario.scenario.scenario_id

        if self.maximum_velocities is None:
            self.maximum_velocities = {}
            network = context.dgscenario.scenario.lanelet_network
            country = context.dgscenario.scenario.scenario_id.country_id
            for lanelet in network.lanelets:
                self.maximum_velocities[lanelet.lanelet_id] = \
                    TrafficSigInterpreter(country=country, lanelet_network=network).speed_limit(
                        frozenset([lanelet.lanelet_id]))

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]
            vehicle_states = traj.values
            network = context.dgscenario.lanelet_network
            states_pos = [np.array([state.x, state.y]) for state in vehicle_states]

            # next step is slow, so approximate by considering entire trajectory inside same lanelet
            # lanelet_ids = network.find_lanelet_by_position(states_pos)
            lanelet_id = network.find_lanelet_by_position([states_pos[0]])
            lanelet_ids = [lanelet_id[0] for i in range(len(states_pos))]
            speed_limits = [self.maximum_velocities[lanelet_id[0]] for lanelet_id in lanelet_ids]

            speed_viol = []
            for i, state in enumerate(vehicle_states):
                max_speed = speed_limits[i]
                if max_speed is None:
                    speed_viol.append(0.0)
                else:
                    if state.vx > max_speed:
                        speed_viol.append(1.0)
                    else:
                        speed_viol.append(0.0)

            ret = self.get_integrated_metric(seq=DgSampledSequence[float](
                values=speed_viol, timestamps=traj.get_sampling_points()))

            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SafeDistance_CR(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This Metric determines if a safe distance between an agent and the leading agent is respected"
    interacting_agents = None
    joint_trajectories = None

    @time_function
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.trajectories[player]

            # todo: WHAT IS t_d?
            def safe_distance(v_l: float, v_e: float, a_e_min: float = -11.25, a_l_min: float = -11.25, t_d=2.0):
                # l: leading vehicle
                # e: ego vehicle
                term_1 = -v_l ** 2 / abs(a_l_min) / 2.0
                term_2 = v_e ** 2 / 2.0 / abs(a_e_min)
                term_3 = t_d * v_e
                return term_1 + term_2 + term_3

            def filter_float(beta: float) -> float:
                if beta < 0.0:
                    return 0.0
                else:
                    return beta

            # convert VehicleState to commonroad State
            ego_state = convert_to_cr_state(traj.values[0], 0)
            # determine what is the leading vehicle (if there is one)
            inter_agents = interacting_agents(scenario=context.dgscenario.scenario,
                                              ego_state=ego_state,
                                              look_ahead_dist=50.0,
                                              around_dist_r=0.0,
                                              around_dist_f=0.0,
                                              around_dist_lat=0.0,
                                              only_leading=True)

            leading_obs = inter_agents["leading"]
            if len(leading_obs) > 0:
                l_name = PlayerName(str(leading_obs[0].obstacle_id))
                l_traj: Trajectory = context.trajectories[l_name]

                dist_viol = [
                    filter_float(safe_distance(v_l=state_l.vx, v_e=state_e.vx)
                                 - np.linalg.norm(np.array([state_l.x, state_l.y]) - np.array([state_e.x, state_e.y])))
                    for state_l, state_e in zip(l_traj.values, traj.values)
                ]

                ret = self.get_integrated_metric(seq=DgSampledSequence[float](
                    values=dist_viol, timestamps=traj.get_sampling_points()))
            else:
                ret = EvaluatedMetric(name=self.get_name(), value=0.0)

            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


# fixme probably all these metrics is better to have them as Type[Metric]?
def get_personal_metrics() -> Set[Metric]:
    metrics: Set[Metric] = {
        EpisodeTime(),
        DeviationLateral(),
        DeviationHeading(),
        DrivableAreaViolation(),
        ProgressAlongReference(),
        LongitudinalAcceleration(),
        LateralComfort(),
        SteeringAngle(),
        SteeringRate(),
    }
    return metrics


def get_joint_metrics() -> Set[Metric]:
    metrics: Set[Metric] = {
        CollisionEnergy(),
        MinimumClearance(),
        ClearanceViolationTime()
    }
    return metrics


# All Metrics
# def get_metrics_set() -> Set[Metric]:
#     metrics: Set[Metric] = get_personal_metrics()
#     metrics |= get_joint_metrics()
#     return metrics


# Only necessary metrics -> speed up computations for Commonroad challenge
def get_metrics_set() -> Set[Metric]:
    metrics: Set[Metric] = {
        SteeringRateSquared(),
        LongitudinalAccelerationSquared(),
        DeviationLateralSquared(),
        RoadCompliance_CR(),
        DistanceToObstacle_CR(),
        ProgressAlongReference(),
        MaximumVelocity_CR(),
        SafeDistance_CR(),
        DrivableAreaViolation()
    }
    return metrics


class MetricEvaluation:
    """Class container to evaluate all the metrics"""

    metrics = get_metrics_set()
    _cache: Dict[JointTrajectories, JointPlayerEvaluatedMetrics] = {}

    def __init__(self):
        raise Exception(f"Don't create instances of {type(self).__name__}!")

    @staticmethod
    def _evaluate_traj(traj: JointTrajectories, world: TrajectoryWorld) -> JointPlayerEvaluatedMetrics:

        if traj in MetricEvaluation._cache:
            return MetricEvaluation._cache[traj]

        context = MetricEvaluationContext(dgscenario=world.scenario, trajectories=traj,
                                          goals=world.goals, geos=world.geo)
        metric_results: Dict[Metric, JointEvaluatedMetric] = {}
        for metric in MetricEvaluation.metrics:
            metric_results[metric] = metric.evaluate(context)

        game_outcome: Dict[PlayerName, PlayerEvaluatedMetrics] = {}
        player_outcome: Dict[Metric, EvaluatedMetric]
        for player in traj.keys():
            player_outcome = {}
            for metric, result in metric_results.items():
                player_outcome[metric] = result[player]
            game_outcome[player] = frozendict(player_outcome)

        ret = frozendict(game_outcome)
        MetricEvaluation._cache[traj] = ret
        return ret

    @staticmethod
    def evaluate(trajectories: JointTrajectories, world: TrajectoryWorld) -> JointPlayerEvaluatedMetrics:
        if trajectories in MetricEvaluation._cache:
            return MetricEvaluation._cache[trajectories]

        outcomes: Dict[PlayerName, Dict[Metric, EvaluatedMetric]] = {}
        traj_step: Dict[PlayerName, Trajectory]
        traj_all: Dict[PlayerName, List[Trajectory]] = {}
        maxl: int = 0
        for player, traj in trajectories.items():
            traj_all[player] = [traj]
            maxl = max(maxl, len(traj_all[player]))

        for i in range(maxl):
            traj_step = {}
            for player in trajectories.keys():
                if i < len(traj_all[player]):
                    traj_step[player] = traj_all[player][i]
            if len(traj_step) == 0:
                break
            step_out = MetricEvaluation._evaluate_traj(traj=frozendict(traj_step), world=world)
            if i == 0:
                for player, out in step_out.items():
                    outcomes[player] = {k: v for k, v in out.items()}
            else:
                for player, all_out in step_out.items():
                    for metric, value in all_out.items():
                        outcomes[player][metric] += value

        ret: Dict[PlayerName, PlayerEvaluatedMetrics] = {}
        for player in outcomes.keys():
            ret[player] = frozendict(outcomes[player])

        frozen_ret = frozendict(ret)
        MetricEvaluation._cache[trajectories] = frozen_ret
        return frozen_ret
