import math
from abc import ABCMeta
from copy import deepcopy
from time import perf_counter
from typing import Tuple, List, Dict, Callable, Set, Mapping
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
    "LongitudinalJerk",
    "LateralComfort",
    "SteeringAngle",
    "SteeringRate",
    "Clearance",
    "CollisionEnergy",
    "MetricEvaluation",
]


def get_evaluated_metric(
    players: List[PlayerName], f: Callable[[PlayerName], EvaluatedMetric]
) -> MetricEvaluationResult:
    mer: Dict[PlayerName, EvaluatedMetric] = {}
    for player_name in players:
        mer[player_name] = f(player_name)
    return mer


def get_values(traj: Trajectory, func: Callable[[VehicleState], float]) \
        -> Tuple[List[Timestamp], List[float]]:
    tval = [(t, func(x)) for t, x in traj]
    interval, val = zip(*tval)
    return interval, val


class EpisodeTime(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "Length of the episode (smaller preferred)"

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            # negative for smaller preferred
            interval = context.get_interval(player)
            val = [1.0 for _ in interval]
            ret = self.get_evaluated_metric(interval=interval, val=val)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationLateral(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the deviation from reference path. "

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            abs_n = [_.distance_from_center for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=abs_n)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationHeading(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric describes the heading deviation from reference path."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            head = [abs(_.relative_heading) for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=head)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DrivableAreaViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the drivable area violation by the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

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

            values = [get_violation(_) for _ in traj_sn]
            ret = self.get_evaluated_metric(interval=interval, val=values)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ProgressAlongReference(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes how far the robot drove **along the reference path** (negative for smaller preferred)"

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            # negative for smaller preferred
            progress = [traj_sn[0].along_lane - _.along_lane for _ in traj_sn]
            inc = differentiate(val=progress, t=interval)
            ret = self.get_evaluated_metric(interval=interval, val=inc)
            if trajectory.p_final is not None:
                ret.total = 100.0       # For finite distance, this metric should be useless
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAcceleration(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the longitudinal acceleration the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, vel = get_values(traj=trajectory, func=get_vel)
            acc = differentiate(vel, interval)
            # Final acc, dacc is zero and not first
            acc_val = [abs(_) for _ in acc[1:]] + [0.0]

            ret = self.get_evaluated_metric(interval=interval, val=acc_val)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalJerk(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the longitudinal acceleration jerk of the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, vel = get_values(traj=trajectory, func=get_vel)
            acc = differentiate(vel, interval)
            dacc = differentiate(acc, interval)
            # Final acc, dacc is zero and not first
            dacc_val = [abs(_) for _ in dacc[1:]] + [0.0]

            ret = self.get_evaluated_metric(interval=interval, val=dacc_val)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LateralComfort(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the lateral discomfort or lateral acceleration the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, ay = get_values(traj=trajectory, func=get_lat_comf)
            ret = self.get_evaluated_metric(interval=interval, val=ay)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringAngle(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the steering angle the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, st = get_values(traj=trajectory, func=get_st)
            st_abs = [abs(_) for _ in st]
            ret = self.get_evaluated_metric(interval=interval, val=st_abs)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRate(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}
    description = "This metric computes the rate of change of steering angle the robot."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, st = get_values(traj=trajectory, func=get_st)
            dst = differentiate(st, interval)
            # Final dst is zero and not first
            dst_val = [abs(_) for _ in dst[1:]] + [0.0]
            ret = self.get_evaluated_metric(interval=interval, val=dst_val)
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class Clearance(Metric, metaclass=ABCMeta):
    PlayersInstance = Mapping[PlayerName, Tuple[SE2Transform, VehicleGeometry]]
    cache_dist: Dict[PlayersInstance, float] = {}
    coeffs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

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
                break

        Clearance.cache_dist[key] = min_dist
        return min_dist


class CollisionEnergy(Clearance):
    cache: Dict[JointPureTraj, Dict[PlayerName, List[float]]] = {}
    cache_joint_traj: Dict[JointPureTraj, Dict[PlayerName, EvaluatedMetric]] = {}
    COLLISION_MIN_DIST = 0.2
    description = "This metric computes the energy of collision between agents."
    time: float = 0.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        tic = perf_counter()

        def calculate_collision(players: List[PlayerName]) -> List[float]:
            assert len(players) == 2
            joint_traj: JointPureTraj = frozendict({p: context.get_trajectory(p) for p in players})
            if joint_traj in self.cache:
                return self.cache[joint_traj][players[0]]
            if players[0] == players[1]:
                energy = [0.0 for _ in context.get_interval(players[0])]
                self.cache[joint_traj] = {players[0]: energy}
                return energy

            L: float = 0.0
            for p in players:
                g = context.get_world().get_geometry(p)
                L += (g.l ** 2 + g.w ** 2) ** 0.5
            energy: List[float] = []
            t1 = list(joint_traj[players[0]].get_sampled_trajectory())
            t2 = list(joint_traj[players[1]].get_sampled_trajectory())
            len1 = min(len(t1), len(t2))
            len2 = max(len(t1), len(t2))
            if len(t1) == len1:
                p1, p2 = players[0], players[1]
            else:
                p2, p1 = players[0], players[1]
            for i in range(len1):
                _, state1 = t1[i]
                _, state2 = t2[i]
                # Coarse collision check
                dx = state1.x - state2.x
                dy = state1.y - state2.y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > L + self.COLLISION_MIN_DIST:
                    energy.append(0.0)
                    continue
                se2_1 = Trajectory.state_to_se2(x=state1)
                se2_2 = Trajectory.state_to_se2(x=state2)
                geo1 = context.get_world().get_geometry(p1)
                geo2 = context.get_world().get_geometry(p2)
                clear_dict = {p1: (se2_1, geo1), p2: (se2_2, geo2)}
                dist = self.get_clearance(players=clear_dict)
                if dist > self.COLLISION_MIN_DIST:
                    energy.append(0.0)
                else:
                    # Calculate energy based on relative velocity between both vehicles
                    vel_proj = math.cos(state1.th - state2.th)
                    vel_relsq = state1.v ** 2 + state2.v ** 2 - 2 * state1.v * state2.v * vel_proj
                    energy_coll = 0.5 * (geo1.m + geo2.m) * vel_relsq
                    energy.append(energy_coll)

            if joint_traj not in self.cache:
                self.cache[joint_traj] = {}
            energy_cp = deepcopy(energy)
            for i in range(len1, len2):
                energy_cp.append(0.0)
            self.cache[joint_traj] = {p1: energy, p2: energy_cp}
            return self.cache[joint_traj][players[0]]

        def calculate_metric(player1: PlayerName) -> EvaluatedMetric:

            joint_traj_all: JointPureTraj = frozendict(
                {p: context.get_trajectory(p) for p in context.get_players()}
            )
            if joint_traj_all in self.cache_joint_traj \
                    and player1 in self.cache_joint_traj[joint_traj_all]:
                return self.cache_joint_traj[joint_traj_all][player1]

            # collision_energy: List[float] = []
            total_energy: float = 0.0
            for player2 in context.get_players():
                coll_e = calculate_collision(players=[player1, player2])
                total_energy += sum(coll_e)
                # if not collision_energy:
                #     collision_energy = coll_e
                # else:
                #     tic_comb = perf_counter()
                #     assert len(collision_energy) == len(coll_e)
                #     collision_energy = [full + val for full, val in zip(collision_energy, coll_e)]
                #     CollisionEnergy.time_comb += perf_counter() - tic_comb

            # TODO[SIR]: Integration is slow, skipping since it's not used
            ret = EvaluatedMetric(title=type(self).__name__, description=self.description,
                                  total=total_energy, incremental=None, cumulative=None)
            # interval = context.get_interval(player1)
            # ret = self.get_evaluated_metric(interval=interval, val=collision_energy)
            if joint_traj_all not in self.cache_joint_traj:
                self.cache_joint_traj[joint_traj_all] = {}
            self.cache_joint_traj[joint_traj_all][player1] = ret
            return ret

        return_val = get_evaluated_metric(context.get_players(), calculate_metric)
        CollisionEnergy.time += perf_counter() - tic
        return return_val


def get_personal_metrics() -> Set[Metric]:
    metrics: Set[Metric] = {
        EpisodeTime(),
        DeviationLateral(),
        DeviationHeading(),
        DrivableAreaViolation(),
        ProgressAlongReference(),
        LongitudinalAcceleration(),
        LongitudinalJerk(),
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
    def evaluate(trajectories: JointPureTraj, world: TrajectoryWorld) -> TrajGameOutcome:

        if trajectories in MetricEvaluation._cache.keys():
            return MetricEvaluation._cache[trajectories]

        context = MetricEvaluationContext(world=world, trajectories=trajectories)
        metric_results: Dict[Metric, MetricEvaluationResult] = {}
        for metric in MetricEvaluation.metrics:
            metric_results[metric] = metric.evaluate(context)

        game_outcome: Dict[PlayerName, PlayerOutcome] = {}
        player_outcome: Dict[Metric, EvaluatedMetric]
        for player in trajectories.keys():
            player_outcome = {}
            for metric, result in metric_results.items():
                player_outcome[metric] = result[player]
            game_outcome[player] = frozendict(player_outcome)

        ret = frozendict(game_outcome)
        MetricEvaluation._cache[trajectories] = ret
        return ret
