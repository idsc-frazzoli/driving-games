from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Tuple, Optional

from duckietown_world import SE2Transform

from dg_commons.planning.lanes import DgLanePose
from dg_commons.seq_op import seq_integrate
from dg_commons.sequence import Timestamp, DgSampledSequence
from games import PlayerName
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld

__all__ = [
    "MetricEvaluationContext",
    "EvaluatedMetric",
    "Metric",
    "PlayerOutcome",
    "MetricEvaluationResult",
    "TrajGameOutcome",
    "differentiate"
]


@dataclass
class MetricEvaluationContext:
    world: TrajectoryWorld
    """ World object. """

    transitions: Mapping[PlayerName, Trajectory]
    """ Trajectory for each player """

    """ Internal data """
    _points_cart: Mapping[PlayerName, List[SE2Transform]] = None
    _points_curv: Mapping[PlayerName, List[DgLanePose]] = None
    """ Sampled vehicle states for each player 
        Cache and reuse for all rules."""

    _cache_cart: Dict[Trajectory, List[SE2Transform]] = None
    _cache_curv: Dict[Trajectory, List[DgLanePose]] = None
    """ Cached transitions to speed up computation, do not set manually """

    def __post_init__(self):
        if MetricEvaluationContext._cache_cart is None:
            MetricEvaluationContext._cache_cart = {}
        if MetricEvaluationContext._cache_curv is None:
            MetricEvaluationContext._cache_curv = {}
        cart: Dict[PlayerName, List[SE2Transform]] = {}
        curv: Dict[PlayerName, List[DgLanePose]] = {}
        for player, trans in self.transitions.items():
            if trans in MetricEvaluationContext._cache_cart.keys():
                cart[player] = MetricEvaluationContext._cache_cart[trans]
                curv[player] = MetricEvaluationContext._cache_curv[trans]
            else:
                traj_cart = trans.get_path_sampled()
                cart[player] = traj_cart
                ref_path = trans.get_lane()
                curv[player] = [ref_path.lane_pose_from_SE2Transform(xy) for xy in traj_cart]
                MetricEvaluationContext._cache_cart[trans] = cart[player]
                MetricEvaluationContext._cache_curv[trans] = curv[player]
        self._points_cart = cart
        self._points_curv = curv

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.transitions[player].get_sampling_points()

    def get_world(self) -> TrajectoryWorld:
        return self.world

    def get_players(self) -> List[PlayerName]:
        return list(self.transitions.keys())

    def get_action(self, player: PlayerName) -> Trajectory:
        return self.transitions[player]

    def get_cartesian_points(self, player: PlayerName) -> List[SE2Transform]:
        return self._points_cart[player]

    def get_curvilinear_points(self, player: PlayerName) -> List[DgLanePose]:
        return self._points_curv[player]


class EvaluatedMetric:
    total: float
    description: str
    title: str
    incremental: DgSampledSequence
    cumulative: DgSampledSequence

    def __init__(
            self,
            title: str,
            description: str,
            total: float,
            incremental: Optional[DgSampledSequence],
            cumulative: Optional[DgSampledSequence],
    ):
        self.title = title
        self.description = description
        self.total = total
        self.incremental = incremental
        self.cumulative = cumulative

    def __repr__(self):
        return f"{self.title} = {self.total:.2f}"

    def __add__(self, other: "EvaluatedMetric") -> "EvaluatedMetric":
        if other is None:
            return self
        return self.add(m1=self, m2=other)

    @staticmethod
    @lru_cache(None)
    def add(m1: "EvaluatedMetric", m2: "EvaluatedMetric") -> "EvaluatedMetric":
        if m1.title != m2.title:
            raise NotImplementedError(f"add implemented only for same metric, "
                                      f"received {m1.title, m2.title}")

        if m1.incremental is None:
            inc = None
        else:
            t_1, t_2 = m1.incremental.timestamps, m2.incremental.timestamps
            if t_1[-1] != t_2[0]:
                raise ValueError(f"Timestamps need to be consecutive - {t_1[-1], t_2[0]}")
            times_i = t_1 + t_2[1:]
            vals_i = m1.incremental.values + m2.incremental.values[1:]
            inc = DgSampledSequence(timestamps=times_i, values=vals_i)

        if m1.cumulative is None:
            cum = None
        else:
            times_c = m1.cumulative.timestamps + m2.cumulative.timestamps
            c_end = m1.cumulative.values[-1]
            vals_c = m1.cumulative.values + [v + c_end for v in m2.cumulative.values]
            cum = DgSampledSequence(timestamps=times_c, values=vals_c)

        return EvaluatedMetric(title=m1.title, description=m1.description,
                               total=m1.total + m2.total, incremental=inc, cumulative=cum)

    __radd__ = __add__


def get_integrated(sequence: DgSampledSequence[float]) -> Tuple[DgSampledSequence[float], float]:
    if len(sequence) <= 1:
        cumulative = 0.0
        dtot = 0.0
    else:
        cumulative = seq_integrate(sequence)
        dtot = cumulative.values[-1]
    return cumulative, dtot


def differentiate(val: List[float], t: List[Timestamp]) -> List[float]:
    if len(val) != len(t):
        msg = "values and times have different sizes - ({},{})," " can't differentiate".format(
            len(val), len(t)
        )
        raise ValueError(msg)

    def func_diff(i: int) -> float:
        dy = val[i + 1] - val[i]
        dx = float(t[i + 1] - t[i])
        if dx < 1e-8:
            raise ValueError(f"identical timestamps for func_diff - {t[i]}")
        return dy / dx

    ret: List[float] = [0.0] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


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
