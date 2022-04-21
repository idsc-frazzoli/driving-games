from itertools import combinations
from typing import Tuple, List, Dict, Callable, Set, Mapping

import geometry as geo
import numpy as np
from commonroad.scenario.traffic_sign import TrafficLight, TrafficLightState
from frozendict import frozendict

from dg_commons import PlayerName, SE2Transform, seq_differentiate, apply_SE2_to_shapely_geo, Timestamp
from dg_commons.maps import DgLanePose, DgLanelet
from dg_commons.planning import Trajectory, JointTrajectories, RefLaneGoal
from dg_commons.seq.sequence import DgSampledSequence
from dg_commons.time import time_function
from driving_games.metrics_structures import (
    EvaluatedMetric,
    JointEvaluatedMetric,
    Metric,
    MetricEvaluationContext,
    PlayerEvaluatedMetrics,
    JointPlayerEvaluatedMetrics,
)
from dg_commons.sim.models.vehicle import VehicleGeometry, VehicleState
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
    "GoalViolation"
]


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
                name=type(self).__name__,
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
    description = "This metric computes the clearance between players (pairwise)."
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
            t_end = float(context.trajectories[pair[0]].get_end())
            assert t_start == context.trajectories[pair[1]].get_start() and \
                   t_end == context.trajectories[pair[1]].get_end(), \
                "The start and end time of different trajectories needs to be the same"

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
    sampling_time: Timestamp = 1.0
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


# todo: refine metric if needed and test
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


def get_metrics_set() -> Set[Metric]:
    metrics: Set[Metric] = get_personal_metrics()
    metrics |= get_joint_metrics()
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
            traj_all[player] = [traj]  # .get_trajectories()
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
