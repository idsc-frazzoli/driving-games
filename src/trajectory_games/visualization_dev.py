from typing import Optional, Mapping, Tuple
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import colorsys
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from dg_commons.planning import JointTrajectories, Trajectory, RefLaneGoal, PlanningGoal

from dg_commons import PlayerName, Color
from driving_games.metrics_structures import MetricEvaluationContext

VehicleObservation = None
VehicleCosts = None
Collision = None


class EvaluationContextVisualization:
    """Visualization for the evaluation context"""

    commonroad_renderer: MPRenderer

    def __init__(
            self,
            evaluation_context: MetricEvaluationContext,
            ax: Axes = None,
            plot_limits: Optional[str] = "auto",
            *args,
            **kwargs
    ):
        self.evaluation_context = evaluation_context
        self.plot_limits = plot_limits
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, figsize=(16, 16), **kwargs)
        self.joint_trajectories: Optional[Mapping[PlayerName, Trajectory]] = evaluation_context.trajectories

    @contextmanager
    def plot_arena_next(self, axis: Axes = None):
        self.commonroad_renderer.ax = axis
        self.commonroad_renderer.f = axis.figure
        self.evaluation_context.dgscenario.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        plt.show()
        yield

    def plot_arena(self, draw_labels: bool, axis: Axes = None):
        if draw_labels:
            self.commonroad_renderer.draw_params["lanelet"]["show_label"] = True
        self.evaluation_context.dgscenario.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        return

    def plot_actions(
            self,
            axis: Axes,
            actions: Mapping[PlayerName, Trajectory],  # question: why use FrozenSet?
            action_colors: Optional[Mapping[PlayerName, Color]] = None,
            goals: Optional[Mapping[PlayerName, PlanningGoal]] = None,
            width: float = 0.7,
            alpha: float = 1.0,
    ) -> Tuple[LineCollection, LineCollection]:
        segments = [np.array([np.array([state.x, state.y]) for state in traj.values]) for _, traj in actions.items()]
        lines = LineCollection(segments=[], colors=[], linewidths=width, alpha=alpha, zorder=ZOrder.actions)
        axis.add_collection(lines)
        lines.set_segments(segments=segments)
        if action_colors is not None:
            colors = [mcolors.to_rgba(action_colors[player]) for player in action_colors]
        else:
            colors = ['blue']
        lines.set_color(colors)

        if goals is not None:
            if isinstance(list(goals.values())[0], RefLaneGoal):
                goal_segments = \
                    [np.array([point.q.p for point in lane.ref_lane.control_points]) for _, lane in goals.items()]

            goal_lines = LineCollection(segments=[], colors=[], linewidths=width, alpha=alpha, zorder=ZOrder.goal)
            axis.add_collection(goal_lines)
            goal_lines.set_segments(segments=goal_segments)
            goal_lines.set_linestyle('--')
            if action_colors is not None:
                goal_colors = [lighten_color(action_colors[player], amount=0.7) for player in action_colors]
            else:
                colors = ['black']
            goal_lines.set_color(goal_colors)

        return lines, goal_lines

    def plot(self, show_plot: bool = False, draw_labels: bool = False,
             action_colors: Optional[Mapping[PlayerName, Color]] = None):
        matplotlib.use('TkAgg')
        self.plot_arena(draw_labels=draw_labels)  # fixme: why does this not show anything with plot_arena_new (yield)
        self.plot_actions(axis=self.commonroad_renderer.ax,
                          actions=self.evaluation_context.trajectories,
                          action_colors=action_colors,
                          goals=self.evaluation_context.goals)
        if show_plot:
            plt.show()


class ZOrder:
    scatter = 60
    lanes = 15
    goal = 20
    actions = 20
    car_box = 75
    player_name = 100


def lighten_color(color: Color, amount: float = 0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
