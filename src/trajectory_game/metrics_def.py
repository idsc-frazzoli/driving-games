from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, List, Mapping
from decimal import Decimal as D

from games import PlayerName
from .sequence import Timestamp, SampledSequence
from .paths import Trajectory
from .world import World

__all__ = [
    "MetricEvaluationContext",
    "EvaluatedMetric",
    "Metric",
    "PlayerOutcome",
    "MetricEvaluationResult",
    "TrajectoryGameOutcome",
]


@dataclass(frozen=True)
class MetricEvaluationContext:
    world: World
    """ World object. """

    trajectories: Mapping[PlayerName, Trajectory]
    """ Sampled vehicle trajectory for each player """

    _points_xy: Mapping[PlayerName, List[Tuple[D, D]]] = None
    _points_sn: Mapping[PlayerName, List[Tuple[D, D]]] = None
    """ Sampled vehicle trajectory for each player 
        Cache and reuse for all rules.
        sn - curvilinear coordinates of player"""

    def __post_init__(self):
        xy: Dict[PlayerName, List[Tuple[D, D]]] = {}
        sn: Dict[PlayerName, List[Tuple[D, D]]] = {}
        for player, traj in self.trajectories.items():
            traj_xy = traj.get_path()
            xy[player] = traj_xy
            ref_path = self.world.get_reference(player)
            sn[player] = ref_path.cartesian_to_curvilinear(traj_xy)

        object.__setattr__(self, "_points_xy", xy)
        object.__setattr__(self, "_points_sn", sn)

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.trajectories[player].get_sampling_points()

    def get_world(self) -> World:
        return self.world

    def get_players(self) -> List[PlayerName]:
        return list(self.trajectories.keys())

    def get_trajectory(self, player: PlayerName) -> Trajectory:
        return self.trajectories[player]

    def get_xy_points(self, player: PlayerName) -> List[Tuple[D, D]]:
        return self._points_xy[player]

    def get_curvilinear_points(self, player: PlayerName) -> List[Tuple[D, D]]:
        return self._points_sn[player]


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
        return f"EvaluatedMetric({self.title}, {self.total})"


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> "MetricEvaluationResult":
        """ Evaluates the metric for all players given a context. """


PlayerOutcome = Mapping[Metric, EvaluatedMetric]
MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
TrajectoryGameOutcome = Mapping[PlayerName, PlayerOutcome]
