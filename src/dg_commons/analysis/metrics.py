from dg_commons.analysis.metrics_def import Metric, MetricEvaluationContext, MetricEvaluationResult, EvaluatedMetric
from trajectory_games.metrics import get_evaluated_metric
from dg_commons import PlayerName
from typing import Union
from sim.models.model_utils import acceleration_constraint
from sim.models.vehicle_utils import steering_constraint
import logging
import matplotlib.pyplot as plt


class Empty:
    pass


class DeviationLateral(Metric):
    brief_description = "deviation_lateral"
    description = "This metric describes the deviation from reference path. "
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval = context.get_interval(player)
            player_lane_pos = context.get_lane_poses(player)

            val = []
            for time in interval:
                lane_pose = player_lane_pos.at(time)
                val.append(float(lane_pose.lateral))

            ret = self.get_evaluated_metric(interval=interval, val=val)
            return ret

        result = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result, context: MetricEvaluationContext, output_dir):

        for player in context.get_players():
            plt.plot(result[player].incremental.timestamps, result[player].incremental.values, label=player)
        self.save_fig(output_dir)

        for i, player in enumerate(context.get_players()):
            center_points = [point.q.p for point in context.planned_lanes[player].control_points]
            x_center = [q[0] for q in center_points]
            y_center = [q[1] for q in center_points]

            x_trajectory = [q.p[0] for q in context.get_poses(player)]
            y_trajectory = [q.p[1] for q in context.get_poses(player)]

            points = context.planned_lanes[player].lane_profile()
            x_lateral = [point[0] for point in points]
            y_lateral = [point[1] for point in points]
            x_max, x_min = max(x_trajectory), min(x_trajectory)
            y_max, y_min = max(y_trajectory), min(y_trajectory)
            delta_x, delta_y = x_max - x_min, y_max - y_min
            if delta_y > delta_x:
                factor_y = 0.1
                factor_x = delta_y/delta_x/4
            else:
                factor_y = delta_x/delta_y/4
                factor_x = 0.1

            plt.xlim(x_min - factor_x * delta_x, x_max + factor_x * delta_x)
            plt.ylim(y_min - factor_y * delta_y, y_max + factor_y * delta_y)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()

            plt.scatter(x=x_trajectory[0], y=y_trajectory[0], c='r', label='x0')
            plt.plot(x_lateral, y_lateral, 'b', label='Boundaries')
            plt.plot(x_center, y_center, '--', linewidth=0.5, color="lightgray", label='Center Line')
            plt.plot(x_trajectory, y_trajectory, 'r', linewidth=0.5, label='Trajectory')

            self.save_fig(output_dir, name=player, dpi=1000)


class DeviationVelocity(Metric):
    brief_description = "deviation_velocity"
    description = "This metric describes the deviation from reference velocity. "
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval = context.get_interval(player)
            player_states = context.actual_trajectory[player]
            target_vels = context.target_velocities[player]

            val = []
            for time in interval:
                player_vel = player_states.at(time).vx
                target_vel = target_vels.at(time)
                val.append(float(abs(player_vel - target_vel)))

            ret = self.get_evaluated_metric(interval=interval, val=val)
            return ret

        result = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result, context: MetricEvaluationContext, output_dir):
        for player in context.get_players():
            plt.plot(result[player].incremental.timestamps, result[player].incremental.values, label=player)

        self.save_fig(output_dir)


class SteeringVelocity(Metric):
    brief_description = "steering_velocity"
    description = "This metric describes the commanded steering velocity"
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval = context.commands[player].timestamps
            commands = context.commands[player]
            states = context.actual_trajectory[player]
            vehicle_params = context.vehicle_params[player]

            val = []
            for time in interval:
                theta = float(states.at(time).delta)
                steering_vel = float(commands.at(time).ddelta)
                logging.disable()
                val.append(steering_constraint(theta, steering_vel, vehicle_params))
                logging.disable(logging.NOTSET)

            ret = self.get_evaluated_metric(interval=interval, val=val)
            return ret

        result = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result, context: MetricEvaluationContext, output_dir):
        for player in context.get_players():
            stamps = result[player].incremental.timestamps
            plt.plot(stamps, result[player].incremental.values, label=player)

        max_value = context.vehicle_params[player].ddelta_max
        min_value = - max_value
        n = len(stamps)
        plt.plot(stamps, n * [max_value], '--', color='lightgray', label='limits')
        plt.plot(stamps, n * [min_value], '--', color='lightgray')

        self.save_fig(output_dir)


class Acceleration(Metric):
    brief_description = "acceleration"
    description = "This metric describes the commanded acceleration"
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval = context.commands[player].timestamps
            commands = context.commands[player]
            states = context.actual_trajectory[player]
            vehicle_params = context.vehicle_params[player]

            val = []
            for time in interval:
                speed = states.at(time).vx
                acc = float(commands.at(time).acc)
                logging.disable()
                val.append(acceleration_constraint(speed, acc, vehicle_params))
                logging.disable(logging.NOTSET)

            ret = self.get_evaluated_metric(interval=interval, val=val)
            return ret

        result = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result, context: MetricEvaluationContext, output_dir):
        for player in context.get_players():
            stamps = result[player].incremental.timestamps
            plt.plot(stamps, result[player].incremental.values, label=player)

        acc_limits = context.vehicle_params[player].acc_limits
        min_value, max_value = acc_limits[0], acc_limits[1]
        n = len(stamps)
        plt.plot(stamps, n * [max_value], '--', color='lightgray', label='limits')
        plt.plot(stamps, n * [min_value], '--', color='lightgray')

        self.save_fig(output_dir)


Metrics = Union[Empty, DeviationVelocity, DeviationLateral, SteeringVelocity, Acceleration]
