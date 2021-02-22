from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping
from decimal import Decimal as D

from duckietown_world import SE2Transform, LanePose

from games import PlayerName
from .sequence import Timestamp, SampledSequence
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld

__all__ = [
    "MetricEvaluationContext",
    "EvaluatedMetric",
    "Metric",
    "PlayerOutcome",
    "MetricEvaluationResult",
    "TrajGameOutcome",
]


@dataclass
class MetricEvaluationContext:
    world: TrajectoryWorld
    """ World object. """

    trajectories: Mapping[PlayerName, Trajectory]
    """ Sampled vehicle trajectory for each player """

    _points_cart: Mapping[PlayerName, List[SE2Transform]] = None
    _points_curv: Mapping[PlayerName, List[LanePose]] = None
    """ Sampled vehicle trajectory for each player 
        Cache and reuse for all rules."""

    _cache_cart: Dict[Trajectory, List[SE2Transform]] = None
    _cache_curv: Dict[Trajectory, List[LanePose]] = None
    """ Cached trajectories to speed up computation, do not set manually """

    def __post_init__(self):
        if MetricEvaluationContext._cache_cart is None:
            MetricEvaluationContext._cache_cart = {}
        if MetricEvaluationContext._cache_curv is None:
            MetricEvaluationContext._cache_curv = {}
        cart: Dict[PlayerName, List[SE2Transform]] = {}
        curv: Dict[PlayerName, List[LanePose]] = {}
        for player, traj in self.trajectories.items():
            if traj in MetricEvaluationContext._cache_cart.keys():
                cart[player] = MetricEvaluationContext._cache_cart[traj]
                curv[player] = MetricEvaluationContext._cache_curv[traj]
            else:
                traj_cart = traj.get_path()
                cart[player] = traj_cart
                ref_path = self.world.get_lane(player)
                curv[player] = [ref_path.lane_pose_from_SE2Transform(xy) for xy in traj_cart]
                MetricEvaluationContext._cache_cart[traj] = cart[player]
                MetricEvaluationContext._cache_curv[traj] = curv[player]
        self._points_cart = cart
        self._points_curv = curv

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.trajectories[player].get_sampling_points()

    def get_world(self) -> TrajectoryWorld:
        return self.world

    def get_players(self) -> List[PlayerName]:
        return list(self.trajectories.keys())

    def get_trajectory(self, player: PlayerName) -> Trajectory:
        return self.trajectories[player]

    def get_cartesian_points(self, player: PlayerName) -> List[SE2Transform]:
        return self._points_cart[player]

    def get_curvilinear_points(self, player: PlayerName) -> List[LanePose]:
        return self._points_curv[player]


class EvaluatedMetric:
    total: D
    description: str
    title: str
    incremental: SampledSequence
    cumulative: SampledSequence

    def __init__(
        self,
        total: D,
        description: str,
        title: str,
        incremental: SampledSequence,
        cumulative: SampledSequence,
    ):
        self.total = total
        self.title = title
        self.incremental = incremental
        self.cumulative = cumulative
        self.description = description

    def __repr__(self):
        return f"{self.title} = {round(float(self.total), 2)}"


class Metric(metaclass=ABCMeta):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        # Allow creation of only one instance of each subclass (singleton)
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Metric, cls).__new__(cls, *args, **kwargs)
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> "MetricEvaluationResult":
        """ Evaluates the metric for all players given a context. """


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
