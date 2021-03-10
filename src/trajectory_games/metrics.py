import math
from typing import Tuple, List, Dict, Mapping, Callable, Set
from decimal import Decimal as D

from duckietown_world import LanePose
from frozendict import frozendict

from games import PlayerName
from .structures import VehicleGeometry, VehicleState
from .sequence import Timestamp, SampledSequence, iterate_with_dt
from .metrics_def import (
    Metric,
    MetricEvaluationContext,
    EvaluatedMetric,
    MetricEvaluationResult,
    TrajGameOutcome,
    PlayerOutcome,
)
from .trajectory_world import TrajectoryWorld
from .paths import Trajectory
from .trajectory_game import JointPureTraj

__all__ = [
    "get_metrics_set",
    "SurvivalTime",
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
    "evaluate_metrics",
]


def integrate(sequence: SampledSequence[float]) -> SampledSequence[float]:
    """ Integrates with respect to time - multiplies the value with delta T. """
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v_avg = (_.v0 + _.v1) / 2.0
        total += v_avg * float(_.dt)
        timestamps.append(Timestamp(_.t0))
        values.append(total)

    return SampledSequence[float](timestamps, values)


def accumulate(sequence: SampledSequence[float]) -> SampledSequence[float]:
    """ Accumulates with respect to time - Sums the values along the horizontal. """
    total = 0.0
    timestamps = []
    values = []
    for t, v in sequence:
        total += v
        timestamps.append(t)
        values.append(total)

    return SampledSequence[float](timestamps, values)


def differentiate(val: List[float], t: List[D]) -> List[float]:
    if len(val) != len(t):
        msg = "values and times have different sizes - ({},{})," " can't differentiate".format(
            len(val), len(t)
        )
        raise ValueError(msg)

    def func_diff(i: int) -> float:
        dy = val[i + 1] - val[i]
        dx = float(t[i + 1] - t[i])
        if dx < 1e-8:
            msg = "identical timestamps for func_diff - {}".format(t[i])
            raise ValueError(msg)
        return dy / dx

    ret: List[float] = [0.0] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


def get_integrated(sequence: SampledSequence[float]) -> Tuple[SampledSequence[float], float]:
    if len(sequence) <= 1:
        cumulative = 0.0
        dtot = 0.0
    else:
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]
    return cumulative, dtot


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


class SurvivalTime(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "Length of the episode (negative for smaller preferred)"

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            traj = trajectory.get_sequence()
            if len(traj) < 1:
                raise ValueError(traj)

            # negative for smaller preferred
            incremental = traj.transform_values(lambda _: -1.0, float)
            cumulative = integrate(incremental)
            total = cumulative.values[-1]

            ret = EvaluatedMetric(
                title=type(self).__name__,
                description=description,
                total=total,
                incremental=incremental,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationLateral(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric describes the deviation from reference path. "

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            abs_n = [_.distance_from_center for _ in traj_sn]
            sequence = SampledSequence[float](interval, abs_n)

            cumulative, dtot = get_integrated(sequence)
            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationHeading(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric describes the heading deviation from reference path."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            head = [abs(_.relative_heading) for _ in traj_sn]

            sequence = SampledSequence[float](interval, head)
            cumulative, dtot = get_integrated(sequence)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DrivableAreaViolation(Metric):
    # TODO[SIR]: This only considers CoG and not car edges
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the drivable area violation by the robot."

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
            sequence = SampledSequence[float](interval, values)
            cumulative, dtot = get_integrated(sequence)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ProgressAlongReference(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes how far the robot drove **along the reference path** (negative for smaller preferred)"

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            # negative for smaller preferred
            progress = [traj_sn[0].along_lane - _.along_lane for _ in traj_sn]
            total = progress[-1]
            inc = [0.0] + [j - i for i, j in zip(progress[:-1], progress[1:])]
            incremental = SampledSequence[float](interval, inc)
            cumulative = SampledSequence[float](interval, progress)

            ret = EvaluatedMetric(
                total=total,
                incremental=incremental,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAcceleration(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the longitudinal acceleration the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, vel = get_values(traj=trajectory, func=get_vel)
            acc = differentiate(vel, interval)
            # Final acc, dacc is zero and not first
            acc_val = [abs(_) for _ in acc[1:]] + [0.0]

            acc_seq = SampledSequence[float](interval, acc_val)
            cumulative, dtot = get_integrated(acc_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=acc_seq,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalJerk(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the longitudinal acceleration jerk of the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, vel = get_values(traj=trajectory, func=get_vel)
            acc = differentiate(vel, interval)
            dacc = differentiate(acc, interval)
            # Final acc, dacc is zero and not first
            dacc_val = [abs(_) for _ in dacc[1:]] + [0.0]

            dacc_seq = SampledSequence[float](interval, dacc_val)
            cumulative, dtot = get_integrated(dacc_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=dacc_seq,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LateralComfort(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the lateral discomfort or lateral acceleration the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, ay = get_values(traj=trajectory, func=get_lat_comf)
            ay_seq = SampledSequence[float](interval, ay)
            cumulative, dtot = get_integrated(ay_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=ay_seq,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringAngle(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the steering angle the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, st = get_values(traj=trajectory, func=get_st)
            st_abs = [abs(_) for _ in st]

            st_seq = SampledSequence[float](interval, st_abs)
            cumulative, dtot = get_integrated(st_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=st_seq,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRate(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the rate of change of steering angle the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval, st = get_values(traj=trajectory, func=get_st)
            dst = differentiate(st, interval)
            # Final dst is zero and not first
            dst_val = [abs(_) for _ in dst[1:]] + [0.0]

            dst_seq = SampledSequence[float](interval, dst_val)
            cumulative, dtot = get_integrated(dst_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=dst_seq,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class CollisionEnergy(Metric):
    cache: Dict[JointPureTraj, List[float]] = {}
    cache_joint_traj: Dict[JointPureTraj, EvaluatedMetric] = {}
    COLLISION_MIN_DIST = 0.2

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        description = "This metric computes the energy of collision between agents."

        def calculate_metric(player1: PlayerName) -> EvaluatedMetric:

            world: TrajectoryWorld = context.get_world()
            geometry: Mapping[PlayerName, VehicleGeometry] = {
                p: world.get_geometry(p) for p in context.get_players()
            }

            def calculate_collision(players: List[PlayerName]) -> List[float]:
                assert len(players) == 2
                joint_traj: JointPureTraj = frozendict({p: context.get_trajectory(p) for p in players})
                if joint_traj in self.cache:
                    return self.cache[joint_traj]

                def get_geo(p_name: PlayerName) -> Tuple[VehicleGeometry, float]:
                    geo: VehicleGeometry = geometry[p_name]
                    dist = (geo.l ** 2 + geo.w ** 2) ** 0.5
                    return geo, dist

                geo1, l1 = get_geo(players[0])
                geo2, l2 = get_geo(players[1])

                energy: List[float] = []
                for (_, state1), (_, state2) in \
                        zip(joint_traj[players[0]].get_sampled_trajectory(),
                            joint_traj[players[1]].get_sampled_trajectory()):

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

                self.cache[joint_traj] = energy
                return energy

            joint_traj_all: JointPureTraj = frozendict(
                {p: context.get_trajectory(p) for p in context.get_players()}
            )
            if joint_traj_all in self.cache_joint_traj:
                return self.cache_joint_traj[joint_traj_all]

            collision_energy: List[float] = []
            for player2 in context.get_players():
                if player1 == player2:
                    timesteps: List[Timestamp] = context.get_interval(player1)
                    coll_e = [0.0 for _ in timesteps]
                else:
                    coll_e = calculate_collision(players=[player1, player2])
                if not collision_energy:
                    collision_energy = coll_e
                else:
                    assert len(collision_energy) == len(coll_e)
                    collision_energy = [full + val for full, val in zip(collision_energy, coll_e)]

            interval = context.get_interval(player1)
            inc = SampledSequence[float](interval, collision_energy)

            cumulative, dtot = get_integrated(inc)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=inc,
                title=type(self).__name__,
                description=description,
                cumulative=cumulative,
            )
            self.cache_joint_traj[joint_traj_all] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


def get_personal_metrics() -> Set[Metric]:
    metrics: Set[Metric] = {
        SurvivalTime(),
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


def evaluate_metrics(
    trajectories: Mapping[PlayerName, Trajectory], world: TrajectoryWorld
) -> TrajGameOutcome:
    metrics: Set[Metric] = get_metrics_set()
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

    return frozendict(game_outcome)
