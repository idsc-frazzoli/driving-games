from functools import lru_cache
from typing import Sequence, Tuple, Mapping, FrozenSet, Optional, Dict, Union

import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager
from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels
from shapely.geometry import Polygon

from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim import Color
from sim.simulator_visualisation import transform_xy
from .game_def import GameVisualization
from .paths import Trajectory
from .preference import PosetalPreference, WeightedPreference
from .structures import VehicleGeometry, VehicleState
from .trajectory_world import TrajectoryWorld

__all__ = ["TrajGameVisualization"]

VehicleObservation = None
VehicleCosts = None
Collision = None


class TrajGameVisualization(GameVisualization[VehicleState, Trajectory, TrajectoryWorld]):
    """ Visualization for the trajectory games"""

    world: TrajectoryWorld
    commonroad_renderer: MPRenderer

    def __init__(self, world: TrajectoryWorld, ax: Axes = None,
                 plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = "auto",
                 *args, **kwargs):
        self.world = world
        self.plot_limits = plot_limits
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, figsize=(16, 16), **kwargs)

    @contextmanager
    def plot_arena(self, axis: Axes):

        self.commonroad_renderer.ax = axis
        self.commonroad_renderer.f = axis.figure
        self.world.scenario.lanelet_network.draw(self.commonroad_renderer,
                                                 draw_params={"traffic_light": {"draw_traffic_lights": False}})
        self.commonroad_renderer.render()
        yield

    def plot_player(self, axis, player_name: PlayerName,
                    state: VehicleState, alpha: float = 0.7, box=None):
        """ Draw the player and his action set at a certain state. """

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        box = plot_car(axis=axis, player_name=player_name,
                       state=state, vg=vg, alpha=alpha, box=box)
        return box

    def plot_equilibria(self, axis, actions: FrozenSet[Trajectory],
                        colour: Color,
                        width: float = .9, alpha: float = 1.0,
                        ticks: bool = True, scatter: bool = True, plot_lanes=True):

        self.plot_actions(axis=axis, actions=actions,
                          colour=colour, width=width,
                          alpha=alpha, ticks=ticks, plot_lanes=plot_lanes)

        if scatter:
            size = (axis.bbox.height / 2000.0) ** 2
            for path in actions:
                vals = [(x.x, x.y, x.v) for _, x in path]
                x, y, vel = zip(*vals)
                scatter = axis.scatter(x, y, s=size, c=vel, marker=".", cmap='winter',
                                       vmin=2.0, vmax=10.0, zorder=ZOrder.scatter)
                # plt.colorbar(scatter, ax=axis)

    def plot_pref(self, axis, pref: PosetalPreference,
                  pname: PlayerName, origin: Tuple[float, float],
                  labels: Mapping[WeightedPreference, str] = None,
                  add_title: bool = True):

        X, Y = origin
        G: DiGraph = pref.graph

        def pos_node(n: WeightedPreference):
            x = G.nodes[n]["x"]
            y = G.nodes[n]["y"]
            return x + X, y + Y

        pos = {_: pos_node(_) for _ in G.nodes}
        text: str
        if labels is None:
            labels = {n: str(n) for n in G.nodes}
            text = "_pref"
        else:
            assert len(G.nodes) == len(labels.keys()), \
                f"Size mismatch between nodes ({len(G.nodes)}) and" \
                f" labels ({len(labels.keys())})"
            for n in G.nodes:
                assert n in labels.keys(), \
                    f"Node {n} not present in keys - {labels.keys()}"
            text = "_outcomes"
        draw_networkx_edges(G, pos=pos, edgelist=G.edges(),
                            ax=axis, arrows=True, arrowstyle="-")

        draw_networkx_labels(G, pos=pos, labels=labels, ax=axis, font_size=8, font_color="b")
        if add_title:
            axis.text(x=X, y=Y + 10.0, s=pname + text, ha="center", va="center")
            axis.set_ylim(top=Y + 15.0)
            # I suspect here we have the problems

    def plot_actions(self, axis: Axes, actions: FrozenSet[Trajectory],
                     colour: Color = None,
                     width: float = .7, alpha: float = 1.0,
                     ticks: bool = True, lines=None, plot_lanes=True) -> LineCollection:
        segments = []
        lanes: Dict[DgLanelet, Optional[Polygon]] = {}
        for traj in actions:
            sampled_traj = np.array([np.array([x.x, x.y]) for _, x in traj])
            segments.append(sampled_traj)
            lane, goal = traj.get_lane()
            lanes[lane] = goal

        for lane, goal in lanes.items():
            points = lane.lane_profile()
            xp, yp = zip(*points)
            x = np.array(xp)
            y = np.array(yp)
            if colour is not None:
                if plot_lanes:
                    axis.fill(x, y, color=colour, alpha=0.2, zorder=ZOrder.lanes)
                #  Toggle to plot goal region
                # if goal is not None:
                #     axis.plot(*goal.exterior.xy, color=colour, linewidth=0.5, zorder=ZOrder.goal)

        if lines is None:
            if colour is None:
                colour = "gray"  # Black
            lines = LineCollection(segments=[],
                                   linewidths=width, alpha=alpha, zorder=ZOrder.actions)

            axis.add_collection(lines)

        lines.set_segments(segments=segments)
        lines.set_color(colour)
        return lines


class ZOrder:
    scatter = 100
    lanes = 15
    goal = 20
    actions = 20
    car_box = 75
    player_name = 100


def plot_car(axis, player_name: PlayerName, state: VehicleState,
             vg: VehicleGeometry, alpha: float, box):
    L = vg.l
    W = vg.w
    car_color = vg.colour
    car: Sequence[Tuple[float, float], ...] = \
        ((-L, -W), (-L, +W), (+L, +W), (+L, -W), (-L, -W))
    xy_theta = (state.x, state.y, state.th)
    q = SE2_from_xytheta(xy_theta)
    car = transform_xy(q, car)
    if box is None:
        box, = axis.fill([], [], color=car_color, edgecolor="saddlebrown", alpha=alpha, zorder=ZOrder.car_box)
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        axis.text(x4+2, y4,
                  player_name,
                  fontsize=9,
                  zorder=ZOrder.player_name,
                  horizontalalignment="left",
                  verticalalignment="center")
    box.set_xy(np.array(car))
    return box


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=.8,
              alpha=.9) -> LineCollection:
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    return LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=ZOrder.actions)


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


@lru_cache
def infer_cmap_from_color(colour):
    colour = colour.lower()
    if colour is None or isinstance(colour, tuple):
        return plt.get_cmap('copper')
    assert isinstance(colour, str)
    if any([w in colour for w in ["red", "fire"]]):
        return plt.get_cmap('YlOrRd')
    if any([w in colour for w in ["green"]]):
        return plt.get_cmap('Greens')
    if any([w in colour for w in ["blue"]]):
        return plt.get_cmap('Blues')
    assert False, colour
