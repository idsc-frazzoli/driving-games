import math
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import partial
from time import perf_counter
from typing import Tuple, List, Dict, Callable, Set, Mapping, Hashable
import numpy as np
import geometry as geo

from duckietown_world import LanePose
from frozendict import frozendict
from duckietown_world import SE2Transform

from games import PlayerName
from .structures import VehicleGeometry, VehicleState
from .sequence import Timestamp
from .metrics_def import (
    Metric,
    MetricEvaluationContext,
    EvaluatedMetric,
    MetricEvaluationResult,
    TrajGameOutcome,
    PlayerOutcome,
    differentiate,
)
from .trajectory_world import TrajectoryWorld
from .paths import Trajectory
from .trajectory_game import JointPureTraj

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
    "MinimumClearance",
    "MetricEvaluation",
]


def get_evaluated_metric(
    players: List[PlayerName], f: Callable[[PlayerName], EvaluatedMetric]
) -> MetricEvaluationResult:
    mer: Dict[PlayerName, EvaluatedMetric] = {}
    for player_name in players:
        mer[player_name] = f(player_name)
    return mer


def get_values(traj: Trajectory, func: Callable[[VehicleState], float], scale: float) \
        -> Tuple[List[Timestamp], List[float]]:
    tval = [(t, func(x)*scale) for t, x in traj]
    interval, val = zip(*tval)
    return interval, val


class EpisodeTime(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "Length of the episode (smaller preferred)"
    scale: float = 2.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval = context.get_interval(player)
            val = [1.0*self.scale for _ in interval]
            ret = self.get_evaluated_metric(interval=interval, val=val)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationLateral(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the deviation from reference path. "
    scale: float = 0.25

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            abs_n = [_.distance_from_center*self.scale for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=abs_n)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationHeading(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the heading deviation from reference path."
    scale: float = 0.5

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            head = [abs(_.relative_heading)*self.scale for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=head)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DrivableAreaViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the drivable area violation by the robot."
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)

            def get_violation(curv: LanePose) -> float:
                diff = 0.0
                if not curv.lateral_inside:
                    if curv.outside_left:
                        diff = curv.distance_from_left
                    elif curv.outside_right:
                        diff = curv.distance_from_right
                return diff

            values = [get_violation(_)*self.scale for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=values)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ProgressAlongReference(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes how far the robot drove **along the reference path** (negative for smaller preferred)"
    scale: float = 0.2

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            # negative for smaller preferred
            progress = [(traj_sn[0].along_lane - p.along_lane) *self.scale for p in traj_sn]
            inc = differentiate(val=progress, t=interval)
            ret = self.get_evaluated_metric(interval=interval, val=inc)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAcceleration(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the longitudinal acceleration the robot."
    scale: float = 0.5

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval, vel = get_values(traj=traj, func=get_vel, scale=self.scale)
            acc = differentiate(vel, interval)
            # Final acc, dacc is zero and not first
            acc_val = [abs(_) for _ in acc[1:]] + [0.0]

            ret = self.get_evaluated_metric(interval=interval, val=acc_val)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LateralComfort(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the lateral discomfort or lateral acceleration the robot."
    scale: float = 0.5

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval, ay = get_values(traj=traj, func=get_lat_comf, scale=self.scale)
            ret = self.get_evaluated_metric(interval=interval, val=ay)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringAngle(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the steering angle the robot."
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval, st = get_values(traj=traj, func=get_st, scale=self.scale)
            st_abs = [abs(_) for _ in st]
            ret = self.get_evaluated_metric(interval=interval, val=st_abs)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRate(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the rate of change of steering angle the robot."
    scale: float = 2.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            traj: Trajectory = context.get_action(player)
            if traj in self.cache:
                return self.cache[traj]

            interval, st = get_values(traj=traj, func=get_st, scale=self.scale)
            dst = differentiate(st, interval)
            # Final dst is zero and not first
            dst_val = [abs(_) for _ in dst[1:]] + [0.0]
            ret = self.get_evaluated_metric(interval=interval, val=dst_val)
            self.cache[traj] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class Clearance(Metric, metaclass=ABCMeta):
    PlayersInstance = Mapping[PlayerName, Tuple[SE2Transform, VehicleGeometry]]
    cache_dist: Dict[PlayersInstance, float] = {}
    coeffs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    THRESHOLD: float
    time: float
    cache_vals: Dict[JointPureTraj, Dict[PlayerName, List[float]]]
    cache_metrics: Dict[JointPureTraj, Dict[PlayerName, EvaluatedMetric]]

    @staticmethod
    def get_clearance(players: PlayersInstance) -> float:
        key = frozendict({name: state for name, (state, _) in players.items()})
        if key in Clearance.cache_dist:
            return Clearance.cache_dist[key]

        players_list = list(players.values())
        min_dist = float("inf")

        for i in range(2):
            p1, geo1 = players_list[i]
            p2, geo2 = players_list[1-i]
            q1, q2 = p1.as_SE2(), p2.as_SE2()
            g = geo.SE2.multiply(geo.SE2.inverse(q1), q2)
            x0, y0, theta = geo.xytheta_from_SE2(g)
            costh, sinth = math.cos(theta), math.sin(theta)
            l, w = geo1.l, geo1.w
            lx, ly = geo2.l*costh, +geo2.l*sinth
            wx, wy = geo2.w*sinth, -geo2.w*costh
            for cl, cw in Clearance.coeffs:
                x = x0 + lx*cl + wx*cw
                y = y0 + ly*cl + wy*cw
                dr = np.array([abs(x)-l, abs(y)-w])
                if all(dr <= 0):
                    min_dist = 0.0
                    break
                elif dr[0] <= 0:
                    min_dist = min(min_dist, dr[1])
                elif dr[1] <= 0:
                    min_dist = min(min_dist, dr[0])
                else:
                    dist = np.linalg.norm(dr)
                    min_dist = min(min_dist, dist)
            if min_dist < 1e-3:
                min_dist = 0.0
                break

        Clearance.cache_dist[key] = min_dist
        return min_dist

    @abstractmethod
    def get_cost(self, dist: float, states: Tuple[VehicleState, VehicleState],
                 geos: Tuple[VehicleGeometry, VehicleGeometry]) -> float:
        """"Calculate cost for given state"""

    @abstractmethod
    def check_threshold(self, dist: float, states: Tuple[VehicleState, VehicleState],
                        geos: Tuple[VehicleGeometry, VehicleGeometry]) -> bool:
        """"Check if value is greater than threshold for given state"""

    def calculate_value(self, context: MetricEvaluationContext,
                        players: List[PlayerName]) -> List[float]:
        assert len(players) == 2
        joint_traj: JointPureTraj = frozendict({p: context.get_action(p) for p in players})
        if joint_traj in self.cache_vals:
            return self.cache_vals[joint_traj][players[0]]
        if players[0] == players[1]:
            values = [0.0 for _ in context.get_interval(players[0])]
            self.cache_vals[joint_traj] = {players[0]: values}
            return values

        L: float = 0.0
        for p in players:
            g = context.get_world().get_geometry(p)
            L += (g.l ** 2 + g.w ** 2) ** 0.5
        values: List[float] = []
        t1, t2 = list(iter(joint_traj[players[0]])), list(iter(joint_traj[players[1]]))
        len1, len2 = min(len(t1), len(t2)), max(len(t1), len(t2))
        if len(t1) == len1:
            p1, p2 = players[0], players[1]
        else:
            p2, p1 = players[0], players[1]
        geo1, geo2 = context.get_world().get_geometry(p1), context.get_world().get_geometry(p2)
        geos = (geo1, geo2)
        for i in range(len1):
            _, state1 = t1[i]
            _, state2 = t2[i]
            states = (state1, state2)
            # Coarse check
            dx = state1.x - state2.x
            dy = state1.y - state2.y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if self.check_threshold(dist=dist-L, states=states, geos=geos):
                values.append(0.0)
                continue
            se2_1 = Trajectory.state_to_se2(x=state1)
            se2_2 = Trajectory.state_to_se2(x=state2)
            clear_dict = {p1: (se2_1, geo1), p2: (se2_2, geo2)}
            dist = self.get_clearance(players=clear_dict)
            values.append(self.get_cost(dist=dist, states=states, geos=geos))

        if joint_traj not in self.cache_vals:
            self.cache_vals[joint_traj] = {}
        values_cp = deepcopy(values)
        for i in range(len1, len2):
            values_cp.append(0.0)
        self.cache_vals[joint_traj] = {p1: values, p2: values_cp}
        return self.cache_vals[joint_traj][players[0]]

    def calculate_metric(self, player1: PlayerName,
                         context: MetricEvaluationContext) -> EvaluatedMetric:

        joint_traj_all: JointPureTraj = frozendict(
            {p: context.get_action(p) for p in context.get_players()}
        )
        if joint_traj_all in self.cache_metrics and player1 in self.cache_metrics[joint_traj_all]:
            return self.cache_metrics[joint_traj_all][player1]

        # all_values: List[float] = []
        total_value: float = 0.0
        for player2 in context.get_players():
            values = self.calculate_value(context=context, players=[player1, player2])
            total_value += sum(values)
            # if not all_values:
            #     all_values = values
            # else:
            #     tic_comb = perf_counter()
            #     assert len(all_values) == len(values)
            #     all_values = [full + val for full, val in zip(all_values, values)]
            #     self.time_comb += perf_counter() - tic_comb

        # TODO[SIR]: Integration is slow, skipping since it's not used
        ret = EvaluatedMetric(title=type(self).__name__, description=self.description,
                              total=total_value, incremental=None, cumulative=None)
        # interval = context.get_interval(player1)
        # ret = self.get_evaluated_metric(interval=interval, val=all_values)
        if joint_traj_all not in self.cache_metrics:
            self.cache_metrics[joint_traj_all] = {}
        self.cache_metrics[joint_traj_all][player1] = ret
        return ret

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        tic = perf_counter()
        metric_func = partial(self.calculate_metric, context=context)
        return_val = get_evaluated_metric(context.get_players(), metric_func)
        self.time += perf_counter() - tic
        return return_val


class CollisionEnergy(Clearance):
    description = "This metric computes the energy of collision between agents."
    time = 0.0
    THRESHOLD = 0.1
    cache_vals: Dict[JointPureTraj, Dict[PlayerName, List[float]]] = {}
    cache_metrics: Dict[JointPureTraj, Dict[PlayerName, EvaluatedMetric]] = {}
    scale: float = 0.01

    def get_cost(self, dist: float, states: Tuple[VehicleState, VehicleState],
                 geos: Tuple[VehicleGeometry, VehicleGeometry]) -> float:
        if self.check_threshold(dist=dist, states=states, geos=geos):
            return 0.0
        # Calculate values based on relative velocity between both vehicles
        state1, state2 = states
        vel_proj = math.cos(state1.th - state2.th)
        vel_relsq = state1.v ** 2 + state2.v ** 2 - 2 * state1.v * state2.v * vel_proj
        energy_coll = vel_relsq * self.scale
        return energy_coll

    def check_threshold(self, dist: float, states: Tuple[VehicleState, VehicleState],
                        geos: Tuple[VehicleGeometry, VehicleGeometry]) -> bool:
        return dist > self.THRESHOLD


class MinimumClearance(Clearance):
    description = "This metric computes the cost when minimum clearance not available between agents."
    time = 0.0
    THRESHOLD = 0.25    # Time between vehicles
    cache_vals: Dict[JointPureTraj, Dict[PlayerName, List[float]]] = {}
    cache_metrics: Dict[JointPureTraj, Dict[PlayerName, EvaluatedMetric]] = {}
    scale: float = 2.0

    def get_cost(self, dist: float, states: Tuple[VehicleState, VehicleState],
                 geos: Tuple[VehicleGeometry, VehicleGeometry]) -> float:
        if self.check_threshold(dist=dist, states=states, geos=geos):
            return 0.0
        return (self.THRESHOLD * max(x.v for x in states) - dist) * self.scale

    def check_threshold(self, dist: float, states: Tuple[VehicleState, VehicleState],
                        geos: Tuple[VehicleGeometry, VehicleGeometry]) -> bool:
        return dist > self.THRESHOLD * max(x.v for x in states)


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


def get_vel(x: VehicleState) -> float:
    return x.v


def get_lat_comf(x: VehicleState) -> float:
    return abs(x.v * x.st)


def get_st(x: VehicleState) -> float:
    return x.st


def get_joint_metrics() -> Set[Metric]:
    metrics: Set[Metric] = {
        CollisionEnergy(),
        MinimumClearance()
    }
    return metrics


def get_metrics_set() -> Set[Metric]:
    metrics: Set[Metric] = get_personal_metrics()
    metrics |= get_joint_metrics()
    return metrics


class MetricEvaluation:
    _cache: Dict[JointPureTraj, TrajGameOutcome] = {}
    metrics = get_metrics_set()

    def __init__(self):
        raise Exception(f"Don't create instances of {type(self).__name__}!")

    @staticmethod
    def _evaluate_traj(traj: JointPureTraj, world: TrajectoryWorld) -> TrajGameOutcome:

        if traj in MetricEvaluation._cache:
            return MetricEvaluation._cache[traj]

        context = MetricEvaluationContext(world=world, transitions=traj)
        metric_results: Dict[Metric, MetricEvaluationResult] = {}
        for metric in MetricEvaluation.metrics:
            metric_results[metric] = metric.evaluate(context)

        game_outcome: Dict[PlayerName, PlayerOutcome] = {}
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
    def evaluate(trajectories: JointPureTraj, world: TrajectoryWorld) -> TrajGameOutcome:
        if not isinstance(trajectories, Hashable):
            trajectories = frozendict(trajectories)

        if trajectories in MetricEvaluation._cache:
            return MetricEvaluation._cache[trajectories]

        outcomes: Dict[PlayerName, Dict[Metric, EvaluatedMetric]] = {}
        traj_step: Dict[PlayerName, Trajectory]
        traj_all: Dict[PlayerName, List[Trajectory]] = {}
        maxl: int = 0
        for player, traj in trajectories.items():
            traj_all[player] = traj.get_trajectories()
            maxl = max(maxl, len(traj_all[player]))

        for i in range(maxl):
            traj_step = {}
            for player in trajectories.keys():
                if i < len(traj_all[player]): traj_step[player] = traj_all[player][i]
            if len(traj_step) == 0: break
            step_out = MetricEvaluation._evaluate_traj(traj=frozendict(traj_step), world=world)
            if i == 0:
                for player, out in step_out.items():
                    outcomes[player] = {k: v for k, v in out.items()}
            else:
                for player, all_out in step_out.items():
                    for metric, value in all_out.items():
                        outcomes[player][metric] += value

        ret: Dict[PlayerName, PlayerOutcome] = {}
        for player in outcomes.keys():
            ret[player] = frozendict(outcomes[player])

        frozen_ret = frozendict(ret)
        MetricEvaluation._cache[trajectories] = frozen_ret
        return frozen_ret
