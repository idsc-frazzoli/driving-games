from typing import Tuple
import textwrap

from .sequence import Timestamp, SampledSequence, iterate_with_dt, UndefinedAtTime
from .rules import Rule, RuleEvaluationContext, RuleEvaluationResult


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


class SurvivalTime(Rule):
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        traj = context.get_trajectory().get_sequence()
        if len(traj) < 1:
            raise ValueError(traj)

        title = "Survival time"
        description = "Length of the episode."

        incremental = traj.transform_values(lambda _: 1.0, float)
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

        if len(sequence) <= 1:
            cumulative = 0
            dtot = 0
        else:
            cumulative = integrate(sequence)
            dtot = cumulative.values[-1]

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
        head = [abs(t-p) for t, p in zip(traj_head, path_head)]

        sequence = SampledSequence[float](interval, head)
        if len(sequence) <= 1:
            cumulative = 0.0
            dtot = 0.0
        else:
            cumulative = integrate(sequence)
            dtot = cumulative.values[-1]

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
                return 0.
            elif v < b[0]:
                return b[0] - v
            else:
                return b[1] - v

        values = [check_bounds(p_n, p_b) for p_n, p_b in zip(s, bounds)]

        sequence = SampledSequence[float](interval, values)
        if len(sequence) <= 1:
            cumulative = 0
            dtot = 0
        else:
            cumulative = integrate(sequence)
            dtot = cumulative.values[-1]

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

        progress = [_-s[0] for _ in s]
        total = progress[-1]
        inc = [0] + [j-i for i, j in zip(progress[:-1], progress[1:])]
        incremental = SampledSequence[float](interval, inc)
        cumulative = SampledSequence[float](interval, progress)

        title = "Reference progress"
        description = textwrap.dedent(
            """\
            This metric computes how far the robot drove
            **along the reference path**.
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
