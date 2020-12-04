from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from .sequence import Timestamp, SampledSequence
from .transitions import Trajectory
from .world import World

__all__ = ["RuleEvaluationContext", "EvaluatedMetric", "RuleEvaluationResult", "Rule"]


@dataclass(frozen=True)
class RuleEvaluationContext:
    # fixme world + all the trajectory game players
    world: World
    """ World object. """

    ego_name: str
    """ Name of the ego-vehicle as an object in the hierarchy """

    trajectory: Trajectory
    """ Sampled vehicle trajectory """

    _points_sn: List[Tuple[float, float]] = None
    """ Sampled vehicle trajectory in curvilinear coordinates 
        Cache and reuse for all rules """

    def __post_init__(self):
        traj_xy = self.trajectory.get_path()
        traj_sn = self.world.ref_path.cartesian_to_curvilinear(traj_xy)
        object.__setattr__(self, "_points_sn", traj_sn)

    def get_interval(self) -> List[Timestamp]:
        return self.trajectory.get_sampling_points()

    def get_world(self) -> World:
        return self.world

    def get_ego_name(self) -> str:
        return self.ego_name

    def get_trajectory(self) -> Trajectory:
        return self.trajectory

    def get_curvilinear_points(self) -> List[Tuple[float, float]]:
        return self._points_sn


class EvaluatedMetric:
    total: float
    description: str
    title: str
    incremental: SampledSequence
    cumulative: SampledSequence

    def __init__(
        self,
        total: float,
        description: str,
        title: str,
        incremental: SampledSequence,
        cumulative: SampledSequence,
    ):
        self.total = float(total)
        self.title = title
        self.incremental = incremental
        self.cumulative = cumulative
        self.description = description

    def __repr__(self):
        return f"EvaluatedMetric({self.title}, {self.total})"


class RuleEvaluationResult:
    # todo this should be more similar to a Mapping[PlayerName,EvaluatedMetric]
    metrics: Dict[Tuple[str, ...], EvaluatedMetric]
    rule: "Rule"

    def __init__(self, rule: "Rule"):
        self.metrics = {}
        self.rule = rule

    def set_metric(
        self,
        name: Tuple[str, ...],
        total: float,
        title: Optional[str] = None,
        description: Optional[str] = None,
        incremental: Optional[SampledSequence] = None,
        cumulative: Optional[SampledSequence] = None,
    ):
        self.metrics[name] = EvaluatedMetric(
            total=total,
            title=title,
            description=description,
            incremental=incremental,
            cumulative=cumulative,
        )

    def __repr__(self):
        return f"RuleEvaluationResult({self.rule}, {self.metrics})"


class Rule(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, context: RuleEvaluationContext, result: RuleEvaluationResult):
        """
        Evaluates the rule in this context.
        Must make at least one call to result.set_violation()
        """
