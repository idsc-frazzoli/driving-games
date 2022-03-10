import colorsys
from contextlib import contextmanager

from geometry import SE2_from_xytheta
from matplotlib import colors as mcolors
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels
from shapely.geometry import Polygon
from commonroad.scenario.scenario import Scenario

from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.metrics_structures import MetricEvaluationContext
from functools import lru_cache
from typing import Tuple, Mapping, Optional, Union, Sequence, FrozenSet
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from dg_commons import Color, transform_xy
from dg_commons import PlayerName
from dg_commons.planning import Trajectory
from trajectory_games.game_def import GameVisualization
from trajectory_games import TrajectoryWorld, PosetalPreference, WeightedMetricPreference

VehicleObservation = None
VehicleCosts = None
Collision = None

#todo [LEON]: merge all these into one general visualizer

class TrajectoryGenerationVisualization:
    """Visualization for commonroad scenario and generated trajectories"""

    commonroad_renderer: MPRenderer

    def __init__(
            self,
            scenario: Scenario,
            ax: Axes = None,
            trajectories: Optional[FrozenSet[Trajectory]] = None,
            plot_limits: Optional[str] = "auto",
            *args,
            **kwargs
    ):
        self.scenario = scenario
        self.plot_limits = plot_limits
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, figsize=(16, 16), **kwargs)
        self.trajectories = trajectories
        self.axis = ax

    def plot_arena(self, draw_labels: bool):
        self.commonroad_renderer.draw_params["trajectory"]["draw_trajectory"] = False
        self.commonroad_renderer.draw_params["dynamic_obstacle"]["draw_shape"] = False
        if draw_labels:
            self.commonroad_renderer.draw_params["lanelet"]["show_label"] = True

        self.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        return

    @staticmethod
    def plot_actions(
            axis: Axes,
            trajectories: FrozenSet[Trajectory],
            color: Color = None) -> LineCollection:
        segments = [np.array([np.array([x.x, x.y]) for _, x in traj]) for traj in trajectories]

        # lines = LineCollection(segments=[], linewidths=width, alpha=alpha, zorder= #old
        lines = LineCollection(segments=[], zorder=ZOrder.actions)
        axis.add_collection(lines)

        lines.set_segments(segments=segments)
        if color is None:
            color = "gray"
        lines.set_color(color)

        return lines

    def plot(
            self,
            show_plot: bool = False,
            draw_labels: bool = False,
            action_color: Optional[Color] = None,
            filename: Optional[str] = None
    ):
        matplotlib.use("TkAgg")
        self.plot_arena(draw_labels=draw_labels)
        if self.trajectories:
            _ = self.plot_actions(
                    axis=self.commonroad_renderer.ax,
                    trajectories=self.trajectories,
                    color=action_color
            )
        if filename:
            plt.savefig(filename)
        if show_plot:
            plt.show()



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

    def plot_arena(self, draw_labels: bool):
        self.commonroad_renderer.draw_params["trajectory"]["draw_trajectory"] = False
        self.commonroad_renderer.draw_params["dynamic_obstacle"]["draw_shape"] = False
        if draw_labels:
            self.commonroad_renderer.draw_params["lanelet"]["show_label"] = True

        self.evaluation_context.dgscenario.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        return

    @staticmethod
    def plot_actions(
            axis: Axes,
            actions: Mapping[PlayerName, Trajectory],
            action_colors: Optional[Mapping[PlayerName, Color]] = None,
            goals: Optional[Mapping[PlayerName, RefLaneGoal]] = None,
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
            colors = ["blue"]
        lines.set_color(colors)

        goal_lines = None
        goal_segments = None
        if goals is not None:
            if isinstance(list(goals.values())[0], RefLaneGoal):
                goal_segments = [
                    np.array([point.q.p for point in lane.ref_lane.get_control_points]) for _, lane in goals.items()
                ]

            goal_lines = LineCollection(segments=[], colors=[], linewidths=width, alpha=alpha, zorder=ZOrder.goal)
            axis.add_collection(goal_lines)
            goal_lines.set_segments(segments=goal_segments)
            goal_lines.set_linestyle("--")
            if action_colors is not None:
                goal_colors = [lighten_color(action_colors[player], amount=0.7) for player in action_colors]
            else:
                goal_colors = ["black"]
            goal_lines.set_color(goal_colors)

        return lines, goal_lines

    def plot(
            self,
            show_plot: bool = False,
            draw_labels: bool = False,
            action_colors: Optional[Mapping[PlayerName, Color]] = None,
    ):
        matplotlib.use("TkAgg")
        self.plot_arena(draw_labels=draw_labels)
        self.plot_actions(
            axis=self.commonroad_renderer.ax,
            actions=self.evaluation_context.trajectories,
            action_colors=action_colors,
            goals=self.evaluation_context.goals,
        )
        if show_plot:
            plt.show()


class TrajGameVisualization(GameVisualization[VehicleState, Trajectory, TrajectoryWorld]):
    """Visualization for the trajectory games"""

    world: TrajectoryWorld
    commonroad_renderer: MPRenderer

    def __init__(
            self,
            world: TrajectoryWorld,
            ax: Axes = None,
            plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = "auto",
            *args,
            **kwargs,
    ):
        self.world = world
        self.plot_limits = "auto"
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, figsize=(16, 16), **kwargs)

    @contextmanager
    def plot_arena(self, axis: Axes):

        self.commonroad_renderer.ax = axis
        self.commonroad_renderer.f = axis.figure
        self.world.scenario.lanelet_network.draw(
            self.commonroad_renderer, draw_params={"traffic_light": {"draw_traffic_lights": False}}
        )
        self.commonroad_renderer.render()
        yield

    def plot_player(self, axis, player_name: PlayerName, state: VehicleState, alpha: float = 0.95, box=None):
        """Draw the player and his action set at a certain state."""

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        box = plot_car(axis=axis, player_name=player_name, state=state, vg=vg, alpha=alpha, box=box)
        return box

    def plot_equilibria(
            self,
            axis,
            actions: FrozenSet[Trajectory],
            colour: Color,
            width: float = 0.9,
            alpha: float = 1.0,
            ticks: bool = True,
            scatter: bool = True,
            plot_lanes=True,
    ):

        self.plot_actions(
            axis=axis, actions=actions, colour=colour, width=width, alpha=alpha, ticks=ticks, plot_lanes=plot_lanes
        )

        if scatter:
            size = np.linalg.norm(axis.bbox.size) / 2000.0
            for path in actions:
                vals = [(x.x, x.y, x.vx) for _, x in path]
                x, y, vel = zip(*vals)
                # scatter = axis.scatter(
                #     x, y, s=size, c=vel, marker=".", cmap="PuRd", vmin=2.0, vmax=10.0, zorder=ZOrder.scatter
                # )
                # plt.colorbar(scatter, ax=axis)

    def plot_pref(
            self,
            axis,
            pref: PosetalPreference,
            pname: PlayerName,
            origin: Tuple[float, float],
            labels: Mapping[WeightedMetricPreference, str] = None,
            add_title: bool = True,
    ):

        X, Y = origin
        G: DiGraph = pref.graph

        def pos_node(n: WeightedMetricPreference):
            x = G.nodes[n]["x"]
            y = G.nodes[n]["y"]
            return x + X, y + Y

        pos = {_: pos_node(_) for _ in G.nodes}
        text: str
        if labels is None:
            labels = {n: str(n) for n in G.nodes}
            text = "_pref"
        else:
            assert len(G.nodes) == len(labels.keys()), (
                f"Size mismatch between nodes ({len(G.nodes)}) and" f" labels ({len(labels.keys())})"
            )
            for n in G.nodes:
                assert n in labels.keys(), f"Node {n} not present in keys - {labels.keys()}"
            text = "_outcomes"
        draw_networkx_edges(G, pos=pos, edgelist=G.edges(), ax=axis, arrows=True, arrowstyle="-")

        draw_networkx_labels(G, pos=pos, labels=labels, ax=axis, font_size=8, font_color="b")
        if add_title:
            axis.text(x=X, y=Y + 10.0, s=pname + text, ha="center", va="center")
            axis.set_ylim(top=Y + 15.0)
            # I suspect here we have the problems

    def plot_actions(
            self,
            axis: Axes,
            actions: FrozenSet[Trajectory],
            lanes: Optional[Mapping[DgLanelet, Optional[Polygon]]] = None,
            colour: Color = None,
            width: float = 0.7,
            alpha: float = 1.0,
            ticks: bool = True,
            lines=None,
            plot_lanes=True,
    ) -> LineCollection:
        # if lanes is None:
        #     lanes: Dict[DgLanelet, Optional[Polygon]] = {}
        #     for traj in actions:
        #         lane, goal = traj.get_lane()
        #         lanes[lane] = goal
        segments = [np.array([np.array([x.x, x.y]) for _, x in traj]) for traj in actions]

        # if plot_lanes and colour is not None:
        #     for lane, goal in lanes.items():
        #         points = lane.lane_profile()
        #         xp, yp = zip(*points)
        #         x = np.array(xp)
        #         y = np.array(yp)
        #         axis.fill(x, y, color=colour, alpha=0.2, zorder=ZOrder.lanes)
        #  Toggle to plot goal region
        # if goal is not None:
        #     axis.plot(*goal.exterior.xy, color=colour, linewidth=0.5, zorder=ZOrder.goal)

        if lines is None:
            if colour is None:
                colour = "gray"  # Black
            lines = LineCollection(segments=[], linewidths=width, alpha=alpha, zorder=ZOrder.actions)
            axis.add_collection(lines)

        lines.set_segments(segments=segments)
        lines.set_color(colour)
        return lines


def plot_car(
        axis, player_name: PlayerName, state: VehicleState, vg: VehicleGeometry, alpha: float, box,
        plot_wheels: bool = True
):
    # L = vg.l
    # W = vg.w
    # todo [LEON]: fix these
    L = vg.lf + vg.lr
    W = vg.w_half * 2
    car_color = vg.color.replace("_car", "")
    car: Sequence[Tuple[float, float], ...] = ((-L, -W), (-L, +W), (+L, +W), (+L, -W), (-L, -W))
    xy_theta = (state.x, state.y, state.theta)
    q = SE2_from_xytheta(xy_theta)
    car = transform_xy(q, car)
    if box is None:
        (vehicle_box,) = axis.fill([], [], color=car_color, edgecolor="saddlebrown", alpha=alpha, zorder=ZOrder.car_box)
        box = [
            vehicle_box,
        ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]

        # axis.text(x4 + 1, y4,
        #           player_name,
        #           fontsize=8,
        #           zorder=ZOrder.player_name,
        #           horizontalalignment="left",
        #           verticalalignment="center")
        if plot_wheels:
            wheels_boxes = [axis.fill([], [], color="k", alpha=alpha, zorder=ZOrder.car_box)[0] for _ in range(4)]
            box.extend(wheels_boxes)
    box[0].set_xy(np.array(car))
    if plot_wheels:
        wheels_outlines = vg.get_rotated_wheels_outlines(state.delta)
        wheels_outlines = [q @ w_outline for w_outline in wheels_outlines]
        for w_idx, wheel in enumerate(box[1:]):
            xy_poly = wheels_outlines[w_idx][:2, :].T
            wheel.set_xy(xy_poly)
    return box


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


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


@lru_cache
def tone_down_color(colour):
    colour = colour.lower()
    if colour == "firebrick":
        return "saddlebrown"
    if colour == "royalblue":
        return "dodgerblue"
    if colour == "forestgreen":
        return "green"
    return colour
