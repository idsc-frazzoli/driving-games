from trajectory_games.metrics_def import EvaluatedMetric, get_integrated
from dg_commons.maps.lanes import DgLanelet
from games.game_def import X, U
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Union, Optional, Dict
from dg_commons import PlayerName
from dg_commons.seq.sequence import Timestamp, DgSampledSequence
from sim.models.vehicle_utils import VehicleParameters
from sim.models.vehicle import VehicleGeometry
from duckietown_world import SE2Transform, LanePose
from geometry import SE2_from_translation_angle
import os
import matplotlib.pyplot as plt


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

    vehicle_params: Optional[Mapping[PlayerName, VehicleParameters]] = None
    """ Vehicle parameters """

    geometry_params: Optional[Mapping[PlayerName, VehicleGeometry]] = None
    """ Vehicle parameters """

    def __post_init__(self):
        self.n_players: int = len(self.planned_lanes.keys())
        """ Number of players """

        self.vehicle_params = dict(zip(self.planned_lanes.keys(), self.n_players*[VehicleParameters.default_car()])) \
            if self.vehicle_params is None else self.vehicle_params

        self.geometry_params = dict(zip(self.planned_lanes.keys(), self.n_players*[VehicleGeometry.default_car()])) \
            if self.geometry_params is None else self.geometry_params

        poses: Dict[PlayerName, List[SE2Transform]] = {}
        lane_poses: Dict[PlayerName, DgSampledSequence[LanePose]] = {}
        for player, Sstate in self.actual_trajectory.items():
            intervals = self.actual_trajectory[player].get_sampling_points()
            helper1 = []
            helper2 = []
            for time in intervals:
                state = Sstate.at(time)
                pose = SE2Transform([state.x, state.y], state.theta)
                lane_pose = self.planned_lanes[player].lane_pose_from_SE2_generic(pose.as_SE2(), 1e-4)
                helper1.append(pose)
                helper2.append(lane_pose)

            poses[player] = helper1
            lane_poses[player] = DgSampledSequence(values=helper2, timestamps=intervals)

        self._pose = poses
        self._lane_pose = lane_poses

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.actual_trajectory[player].get_sampling_points()

    def get_players(self) -> List[PlayerName]:
        return list(self.actual_trajectory.keys())

    def get_poses(self, player: PlayerName) -> List[SE2Transform]:
        return self._pose[player]

    def get_lane_poses(self, player: PlayerName) -> DgSampledSequence[LanePose]:
        return self._lane_pose[player]


class Metric(metaclass=ABCMeta):
    _instances = {}
    brief_description: str
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

    def save_fig(self, output_dir, name=None, dpi=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

        if name is not None:
            fig_file = os.path.join(output_dir, name)
            plt.title(name)
        else:
            fig_file = os.path.join(output_dir, self.brief_description)
            plt.title(self.description)

        if dpi:
            plt.savefig(fig_file, dpi=dpi)
        else:
            plt.savefig(fig_file)

        plt.clf()


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
