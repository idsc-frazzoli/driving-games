from typing import Tuple, List, Dict, Mapping, Callable, Set
from decimal import Decimal as D

from frozendict import frozendict

from games import PlayerName
from .sequence import Timestamp, SampledSequence, iterate_with_dt
from .metrics_def import (
    Metric,
    MetricEvaluationContext,
    EvaluatedMetric,
    MetricEvaluationResult,
    TrajGameOutcome,
    PlayerOutcome,
)
from .world import World
from .paths import Trajectory

# TODO[SIR]: Add __all__


def integrate(sequence: SampledSequence[Timestamp]) -> SampledSequence[Timestamp]:
    """ Integrates with respect to time - multiplies the value with delta T. """
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    total = D("0")
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v0 = D(_.v0)
        dt = _.dt
        total += D(v0 * dt)

        timestamps.append(Timestamp(_.t0))
        values.append(total)

    return SampledSequence[D](timestamps, values)


def accumulate(sequence: SampledSequence[D]) -> SampledSequence[D]:
    """ Accumulates with respect to time - Sums the values along the horizontal. """
    total = 0.0
    timestamps = []
    values = []
    for t, v in sequence:
        total += v
        timestamps.append(t)
        values.append(total)

    return SampledSequence[D](timestamps, values)


def differentiate(val: List[D], t: List[D]) -> List[D]:
    if len(val) != len(t):
        msg = "values and times have different sizes - ({},{})," " can't differentiate".format(
            len(val), len(t)
        )
        raise ValueError(msg)

    def func_diff(i: int) -> D:
        dy = val[i + 1] - val[i]
        dx = t[i + 1] - t[i]
        if dx < 1e-8:
            msg = "identical timestamps for func_diff - {}".format(t[i])
            raise ValueError(msg)
        return dy / dx

    ret: List[D] = [D("0")] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


def get_integrated(sequence: SampledSequence[Timestamp]) -> Tuple[SampledSequence[Timestamp], D]:
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


# TODO[SIR]: Compute only incremental metrics for each edge,
#  cumulative and total only for each trajectory
class SurvivalTime(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Survival time"
        description = "Length of the episode (negative for smaller preferred)"

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            traj = trajectory.get_sequence()
            if len(traj) < 1:
                raise ValueError(traj)

            # negative for smaller preferred
            incremental = traj.transform_values(lambda _: D("-1"), D)
            cumulative = integrate(incremental)
            total = cumulative.values[-1]

            ret = EvaluatedMetric(
                title=title,
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

        title = "Deviation from reference path"
        description = "This metric describes the deviation from reference path. "

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            _, n = zip(*traj_sn)
            abs_n = [abs(_) for _ in n]
            sequence = SampledSequence[D](interval, abs_n)

            cumulative, dtot = get_integrated(sequence)
            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DeviationHeading(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:
        title = "Heading Deviation"
        description = "This metric describes the heading deviation from reference path."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            s, _ = zip(*traj_sn)
            path_head = context.get_world().get_reference(player).heading_at_s(s)
            traj_head = [x.th for _, x in context.get_trajectory(player)]
            head = [abs(t - p) for t, p in zip(traj_head, path_head)]

            sequence = SampledSequence[D](interval, head)
            cumulative, dtot = get_integrated(sequence)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class DrivableAreaViolation(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Drivable area violation"
        description = "This metric computes the drivable area violation by the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            s, n = zip(*traj_sn)
            bounds = context.get_world().get_reference(player).get_bounds_at_s(s)

            def check_bounds(v: D, b: Tuple[D, D]) -> D:
                if b[0] <= v <= b[1]:
                    return D("0")
                elif v < b[0]:
                    return b[0] - v
                else:
                    return b[1] - v

            values = [check_bounds(p_n, p_b) for p_n, p_b in zip(n, bounds)]

            sequence = SampledSequence[D](interval, values)
            cumulative, dtot = get_integrated(sequence)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=sequence,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class ProgressAlongReference(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Reference progress"
        description = "This metric computes how far the robot drove **along the reference path** (negative for smaller preferred)"

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            traj_sn = context.get_curvilinear_points(player)
            s, _ = zip(*traj_sn)

            # negative for smaller preferred
            progress: List[D] = [s[0] - _ for _ in s]
            total: D = progress[-1]
            inc: List[D] = [D("0")] + [j - i for i, j in zip(progress[:-1], progress[1:])]
            incremental = SampledSequence[D](interval, inc)
            cumulative = SampledSequence[D](interval, progress)

            ret = EvaluatedMetric(
                total=total,
                incremental=incremental,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalAcceleration(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Longitudinal acceleration"
        description = "This metric computes the longitudinal acceleration the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]
            interval = context.get_interval(player)

            vel = [x.v for _, x in trajectory]
            acc = differentiate(vel, interval)
            # Final acc, dacc is zero and not first
            acc_val = [abs(_) for _ in acc[1:]] + [D("0")]

            acc_seq = SampledSequence[D](interval, acc_val)
            cumulative, dtot = get_integrated(acc_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=acc_seq,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LongitudinalJerk(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Longitudinal acceleration jerk"
        description = "This metric computes the longitudinal acceleration jerk of the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)

            vel = [x.v for _, x in trajectory]
            acc = differentiate(vel, interval)
            # TODO[SIR]: Improve dacc calc -> Use initial acc
            dacc = differentiate(acc, interval)
            # Final acc, dacc is zero and not first
            dacc_val = [abs(_) for _ in dacc[1:]] + [D("0")]

            dacc_seq = SampledSequence[D](interval, dacc_val)
            cumulative, dtot = get_integrated(dacc_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=dacc_seq,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class LateralComfort(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Lateral discomfort"
        description = "This metric computes the lateral discomfort or lateral acceleration the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)

            ay = [abs(x.v * x.st) for _, x in trajectory]
            ay_seq = SampledSequence[D](interval, ay)
            cumulative, dtot = get_integrated(ay_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=ay_seq,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringAngle(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Steering angle"
        description = "This metric computes the steering angle the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            st = [x.st for _, x in trajectory]
            st_abs = [abs(_) for _ in st]

            st_seq = SampledSequence[D](interval, st_abs)
            cumulative, dtot = get_integrated(st_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=st_seq,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


class SteeringRate(Metric):
    cache: Dict[Trajectory, EvaluatedMetric] = {}

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        title = "Steering rate"
        description = "This metric computes the rate of change of steering angle the robot."

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            trajectory: Trajectory = context.get_trajectory(player)
            if trajectory in self.cache:
                return self.cache[trajectory]

            interval = context.get_interval(player)
            st = [x.st for _, x in trajectory]
            dst = differentiate(st, interval)
            # Final dst is zero and not first
            dst_val = [abs(_) for _ in dst[1:]] + [D("0")]

            dst_seq = SampledSequence[D](interval, dst_val)
            cumulative, dtot = get_integrated(dst_seq)

            ret = EvaluatedMetric(
                total=dtot,
                incremental=dst_seq,
                title=title,
                description=description,
                cumulative=cumulative,
            )
            self.cache[trajectory] = ret
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)


def get_metrics_set() -> Set[Metric]:
    if not get_metrics_set.metrics:
        get_metrics_set.metrics = {
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
    return get_metrics_set.metrics


# fixme this looks weird
get_metrics_set.metrics: Set[Metric] = set()


def evaluate_metrics(trajectories: Mapping[PlayerName, Trajectory], world: World) -> TrajGameOutcome:
    metrics: Set[Metric] = world.metrics
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
