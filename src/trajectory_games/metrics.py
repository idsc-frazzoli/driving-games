import math
from copy import deepcopy
from typing import Tuple, List, Dict, Callable, Set

from duckietown_world import LanePose
from frozendict import frozendict

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


class CollisionEnergy(Metric):
    cache: Dict[JointPureTraj, Dict[PlayerName, List[float]]] = {}
    cache_joint_traj: Dict[JointPureTraj, Dict[PlayerName, EvaluatedMetric]] = {}
    COLLISION_MIN_DIST = 0.2
    description = "This metric computes the energy of collision between agents."

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def get_geo(p_name: PlayerName) -> Tuple[VehicleGeometry, float]:
            geo: VehicleGeometry = context.get_world().get_geometry(p_name)
            dist = (geo.l ** 2 + geo.w ** 2) ** 0.5
            return geo, dist

        def calculate_collision(players: List[PlayerName]) -> List[float]:
            assert len(players) == 2
            joint_traj: JointPureTraj = frozendict({p: context.get_trajectory(p) for p in players})
            if joint_traj in self.cache:
                return self.cache[joint_traj][players[0]]
            if players[0] == players[1]:
                energy = [0.0 for _ in context.get_interval(players[0])]
                self.cache[joint_traj] = {players[0]: energy}
                return energy

            geo1, l1 = get_geo(players[0])
            geo2, l2 = get_geo(players[1])

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
                if dist > l1 + l2:
                    energy.append(0.0)
                    continue

                # Exact collision check
                th_diff = math.atan2(dy, dx)

                def get_projection(state: VehicleState, geo: VehicleGeometry) -> float:
                    # Get the projected distance of the corner of the car along the line joining both car CoGs
                    th_proj = state.th - th_diff
                    cos_proj = abs(math.cos(th_proj))
                    sin_proj = abs(math.sin(th_proj))
                    return geo.l * cos_proj + geo.w * sin_proj

                # If the sum of both projections is smaller than the distance, cars don't collide
                # TODO[SIR]: This is wrong, fix!
                proj1 = get_projection(state1, geo1)
                proj2 = get_projection(state2, geo2)
                if proj1 + proj2 - dist < self.COLLISION_MIN_DIST:
                    energy.append(0.0)
                else:
                    # Calculate energy based on relative velocity between both cars
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

            collision_energy: List[float] = []
            for player2 in context.get_players():
                coll_e = calculate_collision(players=[player1, player2])
                if not collision_energy:
                    collision_energy = coll_e
                else:
                    assert len(collision_energy) == len(coll_e)
                    collision_energy = [full + val for full, val in zip(collision_energy, coll_e)]

            interval = context.get_interval(player1)
            ret = self.get_evaluated_metric(interval=interval, val=collision_energy)
            if joint_traj_all not in self.cache_joint_traj:
                self.cache_joint_traj[joint_traj_all] = {}
            self.cache_joint_traj[joint_traj_all][player1] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


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

    def __init__(self):
        raise Exception(f"Don't create instances of {type(self).__name__}!")

    @staticmethod
    def evaluate(trajectories: JointPureTraj, world: TrajectoryWorld) -> TrajGameOutcome:

        if trajectories in MetricEvaluation._cache.keys():
            return MetricEvaluation._cache[trajectories]

        metrics = get_metrics_set()
        context = MetricEvaluationContext(world=world, trajectories=trajectories)

        metric_results: Dict[Metric, MetricEvaluationResult] = {}
        for metric in metrics:
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
