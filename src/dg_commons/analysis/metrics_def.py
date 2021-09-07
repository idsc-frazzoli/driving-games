from trajectory_games.metrics_def import EvaluatedMetric, get_integrated, differentiate
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Tuple, Optional, TypeVar

from duckietown_world import SE2Transform, LanePose

from dg_commons.seq_op import seq_integrate
from dg_commons.sequence import Timestamp, DgSampledSequence
from games import PlayerName
from dg_commons.planning.lanes import DgLanelet, LaneCtrPoint
from games.game_def import X, U


@dataclass
class MetricEvaluationContext:
    planned_lanes: Mapping[PlayerName, DgLanelet]
    """ Planned lanes to follow for each player """

    actual_trajectory: Mapping[PlayerName, DgSampledSequence[X]]
    """ Trajectory for each player """

    commands: Mapping[PlayerName, DgSampledSequence[U]]
    """ Commands for each player """

    target_velocities: Mapping[PlayerName, DgSampledSequence[float]]
    """ Planned velocities """

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.actual_trajectory[player].get_sampling_points()

    def get_players(self) -> List[PlayerName]:
        return list(self.actual_trajectory.keys())


class Metric(metaclass=ABCMeta):
    _instances = {}
    description: str
    scale: float

    def __new__(cls, *args, **kwargs):
        # Allow creation of only one instance of each subclass (singleton)
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Metric, cls).__new__(cls, *args, **kwargs)
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> "MetricEvaluationResult":
        """ Evaluates the metric for all players given a context. """

    def get_evaluated_metric(self, interval: List[Timestamp], val: List[float]) -> EvaluatedMetric:
        incremental = DgSampledSequence[float](interval, val)
        cumulative, total = get_integrated(incremental)
        ret = EvaluatedMetric(title=type(self).__name__, description=self.description,
                              total=total, incremental=incremental, cumulative=cumulative)
        return ret


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
