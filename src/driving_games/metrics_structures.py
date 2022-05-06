from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping, MutableMapping, Optional, NewType

from dg_commons import fd, PlayerName, SE2Transform, seq_integrate, valmap
from dg_commons.maps import DgLanePose
from dg_commons.planning import JointTrajectories, PlanningGoal, RefLaneGoal
from dg_commons.seq.sequence import DgSampledSequence
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario

__all__ = [
    "MetricEvaluationContext",
    "EvaluatedMetric",
    "JointEvaluatedMetric",
    "Metric",
    "PlayerEvaluatedMetrics",
    "JointPlayerEvaluatedMetrics",
    "MetricNodeName",
    "JointPlayerOutcome",
    "PlayerOutcome",
]

MetricNodeName = NewType("MetricNodeName", str)
"""The metric node name. This corresponds to a unique identifier for the posetal preference graph."""


@dataclass(frozen=True)
class EvaluatedMetric:
    name: MetricNodeName
    value: float
    """ Total value of the metric cost.
    It is usually the min/max/avg/integral/cumsum of the pointwise evaluation of the metric"""
    pointwise: Optional[DgSampledSequence] = None

    def __repr__(self):
        return f"{self.name} = {self.value:.2f}"


JointEvaluatedMetric = Mapping[PlayerName, EvaluatedMetric]


@dataclass
class MetricEvaluationContext:
    dgscenario: DgScenario
    trajectories: JointTrajectories
    goals: Mapping[PlayerName, List[PlanningGoal]]
    """ First PlanningGoal is players' preferred one. If there are other goals, they are equivalent or worse."""
    geos: Mapping[PlayerName, VehicleGeometry]

    """ Cached data for each player use for all rules. """
    points_cart: Mapping[PlayerName, List[SE2Transform]] = field(init=False, default_factory=dict)
    points_curv: Mapping[PlayerName, List[DgLanePose]] = field(init=False, default_factory=dict)

    # _cache_cart: Dict[Trajectory, List[SE2Transform]] = None
    # _cache_curv: Dict[Trajectory, List[DgLanePose]] = None
    # """ Cached transitions to speed up computation, do not set manually """

    def __post_init__(self):
        # cartesian path
        self.points_cart = valmap(lambda x: x.as_path(), self.trajectories)
        # precompute curvilinear coordinates of preferred goal for all players the ones that have a ref lane
        curv: MutableMapping[PlayerName, List[DgLanePose]] = dict()
        for p, ref_lane in self.goals.items():
            if isinstance(ref_lane[0], RefLaneGoal):
                curv[p] = [ref_lane[0].ref_lane.lane_pose_from_SE2Transform(q) for q in self.points_cart[p]]
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
        else:
            RuntimeError("You can only create one instance of each metric")
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> JointEvaluatedMetric:
        """Evaluates the metric for all players given a context."""

    def get_integrated_metric(self, seq: DgSampledSequence[float]) -> EvaluatedMetric:
        tot_value = seq_integrate(seq).values[-1]

        ret = EvaluatedMetric(
            name=self.get_name(),
            value=tot_value,
            pointwise=seq,
        )
        return ret

    def get_metric(self, seq: DgSampledSequence[float]) -> EvaluatedMetric:
        tot_value = seq.values[-1]
        ret = EvaluatedMetric(
            name=self.get_name(),
            value=tot_value,
            pointwise=seq,
        )
        return ret

    def get_name(self) -> MetricNodeName:
        return MetricNodeName(type(self).__name__)


PlayerEvaluatedMetrics = Mapping[Metric, EvaluatedMetric]  # PlayerOutcome
JointPlayerEvaluatedMetrics = Mapping[PlayerName, PlayerEvaluatedMetrics]  # TrajGameOutcome

# MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
JointPlayerOutcome = Mapping[PlayerName, PlayerEvaluatedMetrics]
# TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
