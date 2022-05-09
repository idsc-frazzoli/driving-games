import colorsys
from contextlib import contextmanager
from dataclasses import asdict

from geometry import SE2_from_xytheta, SE2value
from matplotlib import colors as mcolors
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels
from matplotlib.patches import Polygon, Circle
from commonroad.scenario.scenario import Scenario

from dg_commons.maps import DgLanelet
from dg_commons.planning import RefLaneGoal, PlanningGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.metrics_structures import MetricEvaluationContext
from functools import lru_cache
from typing import Tuple, Mapping, Optional, Union, Sequence, FrozenSet, List, Dict, Set
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
from trajectory_games import TrajectoryWorld, PosetalPreference, MetricNodePreference

VehicleObservation = None
VehicleCosts = None
Collision = None


# todo [LEON]: merge all these into one general visualizer


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
            **kwargs,
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
    def plot_actions(axis: Axes, trajectories: FrozenSet[Trajectory], color: Color = None) -> LineCollection:
        segments = [np.array([np.array([x.x, x.y]) for _, x in traj]) for traj in trajectories]

        # lines = LineCollection(segments=[], linewidths=width, alpha=alpha, zorder= #old
        lines = LineCollection(segments=[], zorder=ZOrder.ACTIONS)
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
            filename: Optional[str] = None,
    ):
        matplotlib.use("TkAgg")
        self.plot_arena(draw_labels=draw_labels)
        if self.trajectories:
            _ = self.plot_actions(axis=self.commonroad_renderer.ax, trajectories=self.trajectories, color=action_color)
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
            **kwargs,
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
            self.commonroad_renderer.draw_params["traffic_sign"]["draw_traffic_signs"] = True

        self.evaluation_context.dgscenario.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        return

    @staticmethod
    def plot_actions(
            axis: Axes,
            actions: Mapping[PlayerName, Trajectory],
            action_colors: Optional[Mapping[PlayerName, Color]] = None,
            goals: Optional[Mapping[PlayerName, List[PlanningGoal]]] = None,
            width: float = 0.7,
            alpha: float = 1.0,
    ) -> Tuple[LineCollection, LineCollection]:
        segments = [np.array([np.array([state.x, state.y]) for state in traj.values]) for _, traj in actions.items()]
        lines = LineCollection(segments=[], colors=[], linewidths=width, alpha=alpha, zorder=ZOrder.ACTIONS)
        axis.add_collection(lines)
        lines.set_segments(segments=segments)
        if action_colors is not None:
            colors = [mcolors.to_rgba(action_colors[player]) for player in action_colors]
        else:
            colors = ["blue"]
        lines.set_color(colors)

        goal_lines = None
        goal_segments = None
        goal_colors = []
        if goals is not None:
            if isinstance(list(goals.values())[0][0], RefLaneGoal):
                goals_list = [lane for _, lane in goals.items()]
                if action_colors is not None:
                    goals_colors = [action_colors[pname] for pname in goals.keys()]
                goal_segments = []
                goal_colors = []
                for i, p_goals in enumerate(goals_list):
                    for lane in p_goals:
                        goal_segments.append(np.array([point.q.p for point in lane.ref_lane.get_control_points]))
                        if action_colors is not None:
                            goal_colors.append(lighten_color(goals_colors[i], amount=0.7))

            goal_lines = LineCollection(segments=[], colors=[], linewidths=width, alpha=alpha, zorder=ZOrder.GOAL)
            axis.add_collection(goal_lines)
            goal_lines.set_segments(segments=goal_segments)
            goal_lines.set_linestyle("--")
            if goal_colors is None:
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

    # world: TrajectoryWorld
    commonroad_renderer: MPRenderer

    def __init__(
            self,
            world: TrajectoryWorld,
            ax: Axes = None,
            plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = None,
            *args,
            **kwargs,
    ):
        self.world = world
        # self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        # self.commonroad_renderer: MPRenderer = MPRenderer(
        #     ax=ax, plot_limits=plot_limits, *args, figsize=(16, 16), **kwargs
        # )
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, figsize=(16, 16), **kwargs)
        self.plot_limits = plot_limits

    # @contextmanager
    # def plot_arena(self):#, axis: Axes):
    #
    #     # self.commonroad_renderer.ax = axis
    #     # self.commonroad_renderer.f = axis.figure
    #     # self.world.scenario.lanelet_network.draw(
    #     #     self.commonroad_renderer, draw_params={"traffic_light": {"draw_traffic_lights": False}}
    #     # )
    #     self.world.scenario.lanelet_network.draw(self.commonroad_renderer)
    #     self.commonroad_renderer.render()
    #     return

    def plot_arena(self, draw_labels: bool, plot_limits=None):
        self.commonroad_renderer.draw_params["trajectory"]["draw_trajectory"] = True
        self.commonroad_renderer.draw_params["dynamic_obstacle"]["draw_shape"] = True

        if draw_labels:
            self.commonroad_renderer.draw_params["lanelet"]["show_label"] = True
            self.commonroad_renderer.draw_params["traffic_sign"]["draw_traffic_signs"] = True
            self.commonroad_renderer.draw_params["dynamic_obstacle"]["show_label"] = True

        if plot_limits is not None:
            self.commonroad_renderer.plot_limits = plot_limits
        self.world.scenario.scenario.draw(self.commonroad_renderer)
        self.commonroad_renderer.render()
        return

    def plot(
            self,
            player_states: Optional[Mapping[PlayerName, VehicleState]] = None,
            player_actions: Optional[Mapping[PlayerName, FrozenSet[Trajectory]]] = None,
            player_refs: Optional[Mapping[PlayerName, RefLaneGoal]] = None,
            player_eqs: Optional[Mapping[PlayerName, Trajectory]] = None,
            show_plot: bool = True,
            filename: str = None,
            # draw_labels: bool = False,
            # action_colors: Optional[Mapping[PlayerName, Color]] = None,
    ):
        matplotlib.use("TkAgg")
        # just plot area surrounding Ego vehicle
        ego_position = np.array([player_states[PlayerName("Ego")].x, player_states[PlayerName("Ego")].y])
        deltax_plot = 50
        deltay_plot = 50
        plot_limits = [[ego_position[0] - deltax_plot, ego_position[0] + deltax_plot],
                       [ego_position[1] - deltay_plot, ego_position[1] + deltay_plot]]

        self.plot_arena(draw_labels=False, plot_limits=plot_limits)  # draw_labels=draw_labels)
        # self.plot_actions(
        #     axis=self.commonroad_renderer.ax,
        #     actions=self.evaluation_context.trajectories,
        #     action_colors=action_colors,
        #     goals=self.evaluation_context.goals,
        # )
        axis = self.commonroad_renderer.ax
        if player_states is not None:
            assert player_states.keys() == self.world.geo.keys(), "Mismatch in players for plotting."
            for pname, pstate in player_states.items():
                self.plot_player(axis=axis, player_name=pname, state=pstate, plot_text=True)

        if player_actions and player_refs is not None:
            assert player_actions.keys() == self.world.geo.keys(), "Mismatch in players for plotting."
            for pname, pactions in player_actions.items():
                if pname == PlayerName("Ego"):
                    lanes_colour = "green"
                else:
                    lanes_colour = "red"
                self.plot_actions(axis=axis, actions=pactions, lanes=player_refs,
                                  colour=lanes_colour)  # todo: add reference lanes

        if player_eqs is not None:
            for pname, peqs in player_eqs.items():
                if pname == PlayerName("Ego"):
                    eq_colour = "darkgreen"
                else:
                    eq_colour = "darkred"
                self.plot_equilibria(axis=axis, actions=peqs, colour=eq_colour)

        if filename is not None:
            plt.savefig(filename, format='pdf')  # , dpi=400)
        if show_plot:
            plt.show()
        plt.close()

    # def plot_player(self, axis, player_name: PlayerName, state: VehicleState, alpha: float = 0.95, box=None):
    #     """Draw the player and his action set at a certain state."""
    #
    #
    #     box = plot_car(axis=axis, state=state, vg=vg, alpha=alpha, box=box)
    #     return box

    def plot_player(
            self,
            axis: Axes,
            player_name: PlayerName,
            state: VehicleState,
            model_poly: Optional[List[Polygon]] = None,
            lights_patches: Optional[List[Circle]] = None,
            alpha: float = 0.6,
            plot_wheels: bool = False,
            plot_lights: bool = False,
            plot_text: bool = False,
    ) -> Tuple[List[Polygon], List[Circle]]:
        """Draw the player the state."""
        # todo make it nicer with a map of plotting functions based on the state type

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        if issubclass(type(state), VehicleState):
            return plot_vehicle(
                ax=axis,
                player_name=player_name,
                state=state,
                vg=vg,
                alpha=alpha,
                vehicle_poly=model_poly,
                lights_patches=lights_patches,
                plot_wheels=plot_wheels,
                plot_ligths=plot_lights,
                plot_text=plot_text,
            )
        else:
            raise RuntimeError

    def plot_equilibria(
            self,
            axis,
            actions: Trajectory,
            colour: Color,
            width: float = 0.9,
            alpha: float = 1.0,
            ticks: bool = True,
            scatter: bool = True,
            plot_lanes=True,
    ):

        if isinstance(actions, Trajectory):
            actions = {actions}
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
            labels: Mapping[MetricNodePreference, str] = None,
            add_title: bool = True,
    ):

        X, Y = origin
        G: DiGraph = pref.graph

        def pos_node(n: MetricNodePreference):
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

    def plot_outcomes(
            self,
            player_outcomes: Dict,
            player_prefs: Mapping[PlayerName, PosetalPreference],
            title_info: str,
            filename: str):

        assert player_outcomes.keys() == player_prefs.keys(), "Keys of outcome and preferences don't match"

        n_players = len(player_prefs.keys())
        fig, ax = plt.subplots(nrows=n_players, ncols=1, figsize=(5, 4 * n_players))

        ax_n = 0
        for pname, p_pref in player_prefs.items():
            p_graph = p_pref.graph

            if n_players > 1:
                axis = ax[ax_n]
            else:
                axis = ax
            plot_pref(axis=axis,
                      pref_graph=p_graph,
                      player_outcomes=player_outcomes,
                      player_name=pname,
                      title_info=title_info
                      )
            ax_n = ax_n + 1

        if filename is not None:
            plt.savefig(filename, format='pdf')  # , dpi=400)

        plt.close()

        return

    def plot_actions(
            self,
            axis: Axes,
            actions: Union[FrozenSet[Trajectory], Set[Trajectory]],
            # lanes: Optional[Mapping[DgLanelet, Optional[Polygon]]] = None,
            lanes: Optional[Mapping[PlayerName, RefLaneGoal]] = None,
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

        if lanes is not None:
            if plot_lanes and colour is not None:
                for pname, lane in lanes.items():
                    ctrl_points = lane[0].ref_lane.get_control_points  # .get_control_points()
                    # a = ctrl_points.get_control_points#.control_points()#.get_control_points()
                    points = [point.q.p for point in ctrl_points]
                    xp, yp = zip(*points)
                    x = np.array(xp)
                    y = np.array(yp)
                    # axis.fill(x, y, color=colour, alpha=0.2, zorder=ZOrder.LANES)
                    if pname == PlayerName("Ego"):
                        ref_colour = "green"
                    else:
                        ref_colour = "red"
                    axis.plot(x, y, color=ref_colour, alpha=0.3, zorder=ZOrder.LANES, linestyle="dashed")
        #  Toggle to plot goal region
        # if goal is not None:
        #     axis.plot(*goal.exterior.xy, color=colour, linewidth=0.5, zorder=ZOrder.goal)

        if lines is None:
            if colour is None:
                colour = "gray"  # Black
            lines = LineCollection(segments=[], linewidths=width, alpha=alpha, zorder=ZOrder.ACTIONS)
            axis.add_collection(lines)

        lines.set_segments(segments=segments)
        lines.set_color(colour)
        return lines


def plot_vehicle(
        ax: Axes,
        player_name: PlayerName,
        state: VehicleState,
        vg: VehicleGeometry,
        alpha: float,
        vehicle_poly: Optional[List[Polygon]] = None,
        lights_patches: Optional[List[Circle]] = None,
        plot_wheels: bool = True,
        plot_ligths: bool = True,
        plot_text: bool = False,
) -> Tuple[List[Polygon], List[Circle]]:
    """"""
    vehicle_outline: Sequence[Tuple[float, float], ...] = vg.outline
    vehicle_color: Color = vg.color.replace("_car", "")
    if player_name == PlayerName("Ego"):
        vehicle_color = "green"
    else:
        vehicle_color = "red"
    q = SE2_from_xytheta((state.x, state.y, state.theta))
    if vehicle_poly is None:
        vehicle_box = ax.fill([], [], color=vehicle_color, alpha=alpha, zorder=ZOrder.MODEL)[0]
        vehicle_poly = [
            vehicle_box,
        ]
        if plot_text:
            x4, y4 = transform_xy(q, ((0, 0),))[0]
            y4 = y4 + 2.0  # offset text #todo generalize this
            ax.text(
                x4,
                y4,
                player_name,
                zorder=ZOrder.PLAYER_NAME,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=6,
            )
        if plot_wheels:
            wheels_boxes = [ax.fill([], [], color="k", alpha=alpha, zorder=ZOrder.MODEL)[0] for _ in range(vg.n_wheels)]
            vehicle_poly.extend(wheels_boxes)
        if plot_ligths:
            lights_patches = _plot_lights(ax=ax, q=q, vg=vg)

    outline = transform_xy(q, vehicle_outline)
    vehicle_poly[0].set_xy(outline)

    if plot_wheels:
        wheels_outlines = vg.get_rotated_wheels_outlines(state.delta)
        wheels_outlines = [q @ w_outline for w_outline in wheels_outlines]
        for w_idx, wheel in enumerate(vehicle_poly[1:]):
            xy_poly = wheels_outlines[w_idx][:2, :].T
            wheel.set_xy(xy_poly)

    if plot_ligths:
        for i, name in enumerate(vg.lights_position):
            position = vg.lights_position[name]
            x2, y2 = transform_xy(q, (position,))[0]
            lights_patches[i].center = x2, y2

    return vehicle_poly, lights_patches


def _plot_lights(ax: Axes, q: SE2value, vg: VehicleGeometry) -> List[Circle]:
    radius_light = 0.04 * vg.width
    patches = []
    for name in vg.lights_position:
        position = vg.lights_position[name]
        x2, y2 = transform_xy(q, (position,))[0]
        patch = Circle((x2, y2), radius=radius_light, zorder=ZOrder.LIGHTS)
        patches.append(patch)
        ax.add_patch(patch)
    return patches


def plot_pref(
        axis,
        pref_graph: DiGraph,
        player_outcomes: Dict,
        title_info: str,
        player_name: PlayerName,
):
    origin = (0, 0)
    X, Y = origin
    G = pref_graph

    def pos_node(n: MetricNodePreference):
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        return x + X, y + Y

    pos = {_: (pos_node(_)) for _ in G.nodes}

    min_x = 1000.0
    max_x = -1000.0
    min_y = 1000.0
    max_y = -1000.0

    for p in pos.values():
        x = p[0]
        y = p[1]
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    labels = {}
    metric_values = {}
    for node in G.nodes:
        c = list(node.weights.keys())
        metric_values[node] = 0.0

        for ci in c:
            metric_values[node] += player_outcomes[player_name][ci].value * node.weights[ci]

        labels[node] = node.name + ": " + '{0:.2f}'.format(metric_values[node])

    draw_networkx_edges(G, pos=pos, edgelist=G.edges(), ax=axis, arrows=True, arrowstyle="-", node_size=700)

    draw_networkx_labels(G, pos=pos, labels=labels, ax=axis, font_size=4, font_color="b")

    axis.text(x=X, y=Y + max_y + 10.0, s=player_name + ", "+ title_info, ha="center", va="center", fontsize=6)
    axis.set_ylim(top=Y + max_y + 20.0, bottom= Y + min_y - 20.0)
    axis.set_xlim(left=X + min_x - 10.0, right=X + max_x + 10.0)


class ZOrder:
    SCATTER = 60
    LANES = 15
    GOAL = 20
    ACTIONS = 20
    CAR_BOX = 75
    PLAYER_NAME = 100
    LIGHTS = 35
    MODEL = 30


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
