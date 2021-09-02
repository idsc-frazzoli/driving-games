from dg_commons.analysis.metrics_def import Metric, MetricEvaluationContext, MetricEvaluationResult, EvaluatedMetric
from trajectory_games.metrics import get_evaluated_metric, get_values
from games import PlayerName
from geometry import SO2value, SO2_from_angle, SE2_from_translation_angle, SE2value, SE2, \
    translation_angle_scale_from_E2, translation_angle_from_SE2, T2value
import numpy as np
from typing import Union


class DeviationLateral(Metric):
    description = "This metric describes the deviation from reference path. "
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext) -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval = context.get_interval(player)
            player_pos = context.actual_trajectory[player]
            dg_traj = context.planned_lanes[player]

            val = []
            for time in interval:
                pose = player_pos.at(time)
                position: np.ndarray = np.array([pose.x, pose.y])
                _, pose_on_lane = dg_traj.find_along_lane_closest_point(position)
                position_on_lane, _ = translation_angle_from_SE2(pose_on_lane)
                val.append(float(np.linalg.norm(position-position_on_lane)))

            ret = self.get_evaluated_metric(interval=interval, val=val)
            return ret

        return get_evaluated_metric(context.get_players(), calculate_metric)
