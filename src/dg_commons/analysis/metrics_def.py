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
from dg_commons import SE2Transform, relative_pose
import os
import matplotlib.pyplot as plt
from geometry import translation_angle_scale_from_E2, SE2_from_translation_angle


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

    betas: Mapping[PlayerName, DgSampledSequence[float]]
    """ Where on the lane """

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

        for player, betas in self.betas.items():
            intervals = self.betas[player].get_sampling_points()
            path = self.planned_lanes[player]
            helper1 = []
            helper2 = []
            for time in intervals:
                beta = betas.at(time)
                state = self.actual_trajectory[player].at(time)

                position, angle = [state.x, state.y], state.theta
                q = SE2_from_translation_angle(position, angle)
                q0 = path.center_point(beta)

                along_lane = path.along_lane_from_beta(beta)
                rel = relative_pose(q, q0)
                r, relative_heading, _ = translation_angle_scale_from_E2(rel)
                lateral = r[1]

                lane_pose = path.lane_pose(along_lane=along_lane, relative_heading=relative_heading, lateral=lateral)

                helper1.append(SE2Transform(position, angle))
                helper2.append(lane_pose)

            poses[player] = helper1
            lane_poses[player] = DgSampledSequence(values=helper2, timestamps=intervals)

            self._pose = poses
            self._lane_pose = lane_poses

    def get_interval(self, player: PlayerName) -> List[Timestamp]:
        return self.actual_trajectory[player].get_sampling_points(), self.commands[player].get_sampling_points()

    def get_players(self) -> List[PlayerName]:
        return list(self.actual_trajectory.keys())

    def get_poses(self, player: PlayerName) -> List[SE2Transform]:
        return self._pose[player]

    def get_lane_poses(self, player: PlayerName) -> DgSampledSequence[LanePose]:
        return self._lane_pose[player]


class Metric(metaclass=ABCMeta):
    _instances = {}
    brief_description: str
    file_name: str
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

    def plot_increment_cumulative(self, result, context: MetricEvaluationContext, output_dir):
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        for player in context.get_players():
            stamps = result[player].incremental.timestamps
            axs[0].plot(stamps, result[player].incremental.values, label=player)
            axs[0].set_title("Absolute " + self.brief_description)
            axs[1].plot(result[player].cumulative.timestamps, result[player].cumulative.values, label=player)
            axs[1].set_title("Integral " + self.brief_description)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)

        for ax in fig.get_axes():
            ax.label_outer()
        Metric.save_fig(output_dir, title="", name=self.file_name + "_absolute", fig=fig)
        return stamps

    @staticmethod
    def save_fig(output_dir, title, name, dpi=None, fig=plt):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)
        try:
            fig.title(title)
        except:
            fig.suptitle(title)
        fig_file = os.path.join(output_dir, name)

        fig.savefig(fig_file, dpi=dpi)
        plt.clf()


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
