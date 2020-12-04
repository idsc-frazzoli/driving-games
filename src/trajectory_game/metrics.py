from typing import Tuple, List, Dict
import textwrap

from .sequence import Timestamp, SampledSequence, iterate_with_dt
from .rules import Rule, RuleEvaluationContext, RuleEvaluationResult
from .world import World
from .transitions import Trajectory


def integrate(sequence: SampledSequence[Timestamp]) -> SampledSequence[Timestamp]:
    """ Integrates with respect to time - multiplies the value with delta T. """
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v0 = float(_.v0)
        dt = _.dt
        total += float(v0 * dt)

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


def differentiate(val: List[float], t: List[float]) -> List[float]:
    if len(val) != len(t):
        msg = "values and times have different sizes - ({},{})," " can't differentiate".format(
            len(val), len(t)
        )
        raise ValueError(msg)

    def func_diff(i: int) -> float:
        dy = val[i + 1] - val[i]
        dx = t[i + 1] - t[i]
        if dx < 1e-8:
            msg = "identical timestamps for func_diff - {}".format(t[i])
            raise ValueError(msg)
        return dy / dx

    ret: List[float] = [0.0] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


def get_integrated(sequence: SampledSequence[Timestamp]) -> Tuple[SampledSequence[Timestamp], float]:
    if len(sequence) <= 1:
        cumulative = 0.0
        dtot = 0.0
    else:
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]
    return cumulative, dtot


# TODO[SIR]: Compute only incremental metrics for each edge,
#  cumulative and total only for each trajectory
class SurvivalTime(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        traj = context.get_trajectory().get_sequence()
        if len(traj) < 1:
            raise ValueError(traj)

        title = "Survival time"
        description = "Length of the episode (negative for smaller preferred)"

        # negative for smaller preferred
        incremental = traj.transform_values(lambda _: -1.0, float)
        cumulative = integrate(incremental)
        total = cumulative.values[-1]

        result.set_metric(
            name=(),
            title=title,
            description=description,
            total=total,
            incremental=incremental,
            cumulative=cumulative,
        )


class DeviationLateral(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        interval = context.get_interval()
        traj_sn = context.get_curvilinear_points()
        _, n = zip(*traj_sn)
        abs_n = [abs(_) for _ in n]
        sequence = SampledSequence[float](interval, abs_n)

        cumulative, dtot = get_integrated(sequence)
        title = "Deviation from reference path"
        description = textwrap.dedent(
            """\
            This metric describes the amount of deviation from the reference path.
        """
        )
        result.set_metric(
            name=(),
            total=dtot,
            incremental=sequence,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class DeviationHeading(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = context.get_interval()
        traj_sn = context.get_curvilinear_points()
        s, _ = zip(*traj_sn)
        path_head = context.get_world().get_heading_at_s(s)
        traj_head = [x.th for _, x in context.trajectory]
        head = [abs(t - p) for t, p in zip(traj_head, path_head)]

        sequence = SampledSequence[float](interval, head)
        cumulative, dtot = get_integrated(sequence)

        title = "Heading Deviation"
        description = textwrap.dedent(
            """\
            This metric describes the amount of deviation from the reference heading.
        """
        )
        result.set_metric(
            name=(),
            total=dtot,
            incremental=sequence,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class DrivableAreaViolation(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = context.get_interval()
        traj_sn = context.get_curvilinear_points()
        s, n = zip(*traj_sn)
        bounds = context.get_world().get_bounds_at_s(s)

        def check_bounds(v: float, b: Tuple[float, float]) -> float:
            if b[0] <= v <= b[1]:
                return 0.0
            elif v < b[0]:
                return b[0] - v
            else:
                return b[1] - v

        values = [check_bounds(p_n, p_b) for p_n, p_b in zip(n, bounds)]

        sequence = SampledSequence[float](interval, values)
        cumulative, dtot = get_integrated(sequence)

        title = "Drivable area violation"
        description = textwrap.dedent(
            """\
            This metric computes the drivable area violation by the robot.
        """
        )

        result.set_metric(
            name=(),
            total=dtot,
            incremental=sequence,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class ProgressAlongReference(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        interval = context.get_interval()
        traj_sn = context.get_curvilinear_points()
        s, _ = zip(*traj_sn)

        # negative for smaller preferred
        progress: List[float] = [s[0] - _ for _ in s]
        total: float = progress[-1]
        inc: List[float] = [0.0] + [j - i for i, j in zip(progress[:-1], progress[1:])]
        incremental = SampledSequence[float](interval, inc)
        cumulative = SampledSequence[float](interval, progress)

        title = "Reference progress"
        description = textwrap.dedent(
            """\
            This metric computes how far the robot drove
            **along the reference path**.
            (negative for smaller preferred)
        """
        )
        result.set_metric(
            name=(),
            total=total,
            incremental=incremental,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class LongitudinalComfort(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = context.get_interval()
        traj = context.get_trajectory()

        vel = [x.v for _, x in traj]
        acc = differentiate(vel, interval)
        # TODO[SIR]: Improve dacc calc -> Use initial acc
        dacc = differentiate(acc, interval)

        # Final acc, dacc is zero and not first
        acc_val = [abs(_) for _ in acc[1:]] + [0.0]
        dacc_val = [abs(_) for _ in dacc[1:]] + [0.0]

        acc_seq = SampledSequence[float](interval, acc_val)
        cumulative, dtot = get_integrated(acc_seq)

        title = "Longitudinal acceleration"
        description = textwrap.dedent(
            """\
            This metric computes the longitudinal acceleration the robot.
        """
        )

        result.set_metric(
            name=("long_acc",),
            total=dtot,
            incremental=acc_seq,
            title=title,
            description=description,
            cumulative=cumulative,
        )

        dacc_seq = SampledSequence[float](interval, dacc_val)
        cumulative, dtot = get_integrated(dacc_seq)

        title = "Longitudinal acceleration jerk"
        description = textwrap.dedent(
            """\
            This metric computes the rate of change of longitudinal acceleration the robot.
        """
        )

        result.set_metric(
            name=("long_jerk",),
            total=dtot,
            incremental=dacc_seq,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class LateralComfort(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = context.get_interval()
        traj = context.get_trajectory()

        ay = [abs(x.v * x.st) for _, x in traj]

        ay_seq = SampledSequence[float](interval, ay)
        cumulative, dtot = get_integrated(ay_seq)

        title = "Lateral discomfort"
        description = textwrap.dedent(
            """\
            This metric computes the lateral discomfort - 
            related to the lateral acceleration the robot.
        """
        )

        result.set_metric(
            name=(),
            total=dtot,
            incremental=ay_seq,
            title=title,
            description=description,
            cumulative=cumulative,
        )


class SteeringComfort(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):

        interval = context.get_interval()
        traj = context.get_trajectory()

        st = [x.st for _, x in traj]
        dst = differentiate(st, interval)

        st_abs = [abs(_) for _ in st]
        # Final dst is zero and not first

        st_seq = SampledSequence[float](interval, st_abs)
        cumulative, dtot = get_integrated(st_seq)

        title = "Steering angle"
        description = textwrap.dedent(
            """\
            This metric computes the steering angle the robot.
        """
        )

        result.set_metric(
            name=("steering",),
            total=dtot,
            incremental=st_seq,
            title=title,
            description=description,
            cumulative=cumulative,
        )

        dst_val = [abs(_) for _ in dst[1:]] + [0.0]
        dst_seq = SampledSequence[float](interval, dst_val)
        cumulative, dtot = get_integrated(dst_seq)

        title = "Steering rate"
        description = textwrap.dedent(
            """\
            This metric computes the rate of change of steering angle the robot.
        """
        )

        result.set_metric(
            name=("steering_rate",),
            total=dtot,
            incremental=dst_seq,
            title=title,
            description=description,
            cumulative=cumulative,
        )


def evaluate_rules(trajectory: Trajectory, world: World, ego_name: str) -> Dict[str, RuleEvaluationResult]:

    rules = {}
    rules["survival_time"] = SurvivalTime()
    rules["deviation-lateral"] = DeviationLateral()
    rules["deviation-heading"] = DeviationHeading()
    rules["drivable-area-violation"] = DrivableAreaViolation()
    rules["progress"] = ProgressAlongReference()
    rules["comfort-longitudinal"] = LongitudinalComfort()
    rules["comfort-lateral"] = LateralComfort()
    rules["comfort-steering"] = SteeringComfort()

    context = RuleEvaluationContext(world=world, ego_name=ego_name, trajectory=trajectory)

    evaluated = {}
    for name, rule in rules.items():
        result = RuleEvaluationResult(rule)
        rule.evaluate(context, result)
        evaluated[name] = result
    return evaluated
