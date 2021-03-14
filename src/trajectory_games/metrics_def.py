from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Mapping, Tuple

from duckietown_world import SE2Transform, LanePose

from games import PlayerName
from .sequence import Timestamp, SampledSequence, iterate_with_dt
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
                traj_cart = traj.get_path_sampled()
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
    total: float
    description: str
    title: str
    incremental: SampledSequence
    cumulative: SampledSequence

    def __init__(
        self,
        title: str,
        description: str,
        total: float,
        incremental: SampledSequence,
        cumulative: SampledSequence,
    ):
        self.title = title
        self.description = description
        self.total = total
        self.incremental = incremental
        self.cumulative = cumulative

    def __repr__(self):
        return f"{self.title} = {self.total:.2f}"


def integrate(sequence: SampledSequence[float]) -> SampledSequence[float]:
    """ Integrates with respect to time - multiplies the value with delta T. """
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    total = 0.0
    timestamps = []
    values = []
    for _ in iterate_with_dt(sequence):
        v_avg = (_.v0 + _.v1) / 2.0
        total += v_avg * float(_.dt)
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
            msg = "identical timestamps for func_diff - {}".format(t[i])
            raise ValueError(msg)
        return dy / dx

    ret: List[float] = [0.0] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


def get_integrated(sequence: SampledSequence[float]) -> Tuple[SampledSequence[float], float]:
    if len(sequence) <= 1:
        cumulative = 0.0
        dtot = 0.0
    else:
        cumulative = integrate(sequence)
        dtot = cumulative.values[-1]
    return cumulative, dtot


class Metric(metaclass=ABCMeta):
    _instances = {}
    description: str

    def __new__(cls, *args, **kwargs):
        # Allow creation of only one instance of each subclass (singleton)
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Metric, cls).__new__(cls, *args, **kwargs)
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> "MetricEvaluationResult":
        """ Evaluates the metric for all players given a context. """

    def get_evaluated_metric(self, interval: List[Timestamp], val: List[float]) -> EvaluatedMetric:
        incremental = SampledSequence[float](interval, val)
        cumulative, total = get_integrated(incremental)
        ret = EvaluatedMetric(title=type(self).__name__, description=self.description,
                              total=total, incremental=incremental, cumulative=cumulative)
        return ret


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
