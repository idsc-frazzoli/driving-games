from abc import abstractmethod, ABC
from collections import Sequence
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, MutableMapping

from commonroad.scenario.scenario import Scenario

from dg_commons import PlayerName, SE2Transform, seq_integrate
from dg_commons import valmap, fd
from dg_commons.maps import DgLanePose
from dg_commons.planning import JointTrajectories, PlanningGoal, RefLaneGoal
from dg_commons.seq.sequence import DgSampledSequence

__all__ = [
    "MetricEvaluationContext",
    "EvaluatedMetric",
    "JointEvaluatedMetric",
    "Metric",
    "PlayerEvaluatedMetrics",
    "JointPlayerEvaluatedMetrics",
]


@dataclass
class EvaluatedMetric:
    name: str
    value: float
    """Total value of the metric cost.
    It is usually the min/max/avg/integral/cumsum of the pointwise evaluation of the metric"""
    pointwise: Optional[DgSampledSequence] = None

    def __repr__(self):
        return f"{self.name} = {self.value:.2f}"


JointEvaluatedMetric = Mapping[PlayerName, EvaluatedMetric]


@dataclass
class MetricEvaluationContext:
    scenario: Scenario
    trajectories: JointTrajectories
    goals: Mapping[PlayerName, PlanningGoal]

    """ Cached data for each player use for all rules. """
    points_cart: Mapping[PlayerName, List[SE2Transform]] = field(init=False)
    points_curv: Mapping[PlayerName, List[DgLanePose]] = field(init=False)

    # _cache_cart: Dict[Trajectory, List[SE2Transform]] = None
    # _cache_curv: Dict[Trajectory, List[DgLanePose]] = None
    # """ Cached transitions to speed up computation, do not set manually """

    def __post_init__(self):
        # cartesian path
        self.points_cart = valmap(lambda x: x.as_path(), self.trajectories)
        # precompute curvilinear coordinates for all the ones that have a ref lane
        curv: MutableMapping[PlayerName, List[DgLanePose]] = dict()
        for p, goal in self.goals.items():
            if isinstance(goal, RefLaneGoal):
                curv[p] = [goal.ref_lane.lane_pose_from_SE2Transform(q) for q in self.points_cart[p]]
        self.points_curv = fd(curv) if curv else None

    def get_players(self) -> List[PlayerName]:
        return list(self.trajectories.keys())


class Metric(ABC):
    _instances = {}
    description: str

    def __new__(cls, *args, **kwargs):
        # Allow creation of only one instance of each subclass (singleton)
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Metric, cls).__new__(cls, *args, **kwargs)
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        """Evaluates the metric for all players given a context."""

    def get_evaluated_metric(self, seq: DgSampledSequence[float]) -> EvaluatedMetric:
        tot_value = seq_integrate(seq).values[-1]
        ret = EvaluatedMetric(
            name=type(self).__name__,
            value=tot_value,
            pointwise=seq,
        )
        return ret


PlayerEvaluatedMetrics = Mapping[Metric, EvaluatedMetric] # PlayerOutcome
JointPlayerEvaluatedMetrics = Mapping[PlayerName, PlayerEvaluatedMetrics] # TrajGameOutcome

# MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
# PlayerOutcome = Mapping[Metric, EvaluatedMetric]
# TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]