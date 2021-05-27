from numbers import Number
from typing import Sequence, Tuple, Mapping, FrozenSet, Set

import numpy as np
import os
from decorator import contextmanager
from duckietown_world import DuckietownMap
from imageio import imread
from matplotlib.collections import LineCollection
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels

from games import PlayerName
from geometry import SE2_from_xytheta

from world import LaneSegmentHashable
from world.map_loading import map_directory, load_driving_game_map
from .structures import VehicleGeometry, VehicleState
from .paths import Trajectory
from .game_def import GameVisualization
from .preference import PosetalPreference, WeightedPreference
from .trajectory_world import TrajectoryWorld

__all__ = ["TrajGameVisualization"]

VehicleObservation = None
VehicleCosts = None
Collision = None


class TrajGameVisualization(GameVisualization[VehicleState, Trajectory, TrajectoryWorld]):
    """ Visualization for the trajectory games"""

    world: TrajectoryWorld
    grid: DuckietownMap

    def __init__(self, world: TrajectoryWorld):
        self.world = world
        self.grid = load_driving_game_map(name=world.map_name)

    @contextmanager
    def plot_arena(self, axis):

        png_path = os.path.join(map_directory, f"{self.world.map_name}.png")
        img = imread(png_path)
        tile_size = self.grid.tile_size
        H = self.grid["tilemap"].H
        W = self.grid["tilemap"].W
        x_size = tile_size * W
        y_size = tile_size * H
        axis.imshow(img, extent=[0, x_size, 0, y_size])
        axis.set_xlim(left=0, right=x_size)
        axis.set_ylim(bottom=0, top=y_size)

        yield

    def plot_player(self, axis, player_name: PlayerName,
                    state: VehicleState, box=None):
        """ Draw the player and his action set at a certain state. """

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        box = plot_car(axis=axis, player_name=player_name,
                       state=state, vg=vg, box=box)
        return box

    def plot_equilibria(self, axis, actions: FrozenSet[Trajectory],
                        colour: VehicleGeometry.COLOUR,
                        width: float = 1.0, alpha: float = 1.0, ticks: bool = True):

        self.plot_actions(axis=axis, actions=actions,
                          colour=colour, width=width,
                          alpha=alpha, ticks=ticks)

        size = (axis.bbox.height/400.0)**2
        for path in actions:
            vals = [(x.x, x.y, x.t) for _, x in path]
            x, y, t = zip(*vals)
            axis.scatter(x, y, s=size, c=t, zorder=10)

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
            assert len(G.nodes) == len(labels.keys()),\
                f"Size mismatch between nodes ({len(G.nodes)}) and" \
                f" labels ({len(labels.keys())})"
            for n in G.nodes:
                assert n in labels.keys(),\
                    f"Node {n} not present in keys - {labels.keys()}"
            text = "_outcomes"
        draw_networkx_edges(G, pos=pos, edgelist=G.edges(),
                            ax=axis, arrows=True, arrowstyle="-")

        draw_networkx_labels(G, pos=pos, labels=labels, ax=axis, font_size=8, font_color="b")
        if add_title:
            axis.text(x=X, y=Y+10.0, s=pname + text, ha="center", va="center")

    def plot_actions(self, axis, actions: FrozenSet[Trajectory],
                     colour: VehicleGeometry.COLOUR = None,
                     width: float = 1.0, alpha: float = 1.0,
                     ticks: bool = True, lines=None) -> LineCollection:
        segments = []
        lanes: Set[LaneSegmentHashable] = set()
        for traj in actions:
            sampled_traj = np.array([np.array([x.x, x.y]) for _, x in traj])
            segments.append(sampled_traj)
            lanes.add(traj.get_lane())

        for lane in lanes:
            points = lane.lane_profile()
            xp, yp = zip(*points)
            x = np.array(xp)
            y = np.array(yp)
            if colour is not None:
                axis.fill(x, y, color=colour, alpha=0.2, zorder=1)

        if lines is None:
            if colour is None:
                colour = (0.0, 0.0, 0.0)    # Black
            lines = LineCollection(segments=[], colors=colour,
                                   linewidths=width, alpha=alpha)
            lines.set_zorder(5)
            axis.add_collection(lines)
            if ticks:
                axis.yaxis.set_ticks_position("left")
                axis.xaxis.set_ticks_position("bottom")
            else:
                axis.yaxis.set_visible(False)
                axis.xaxis.set_visible(False)
        lines.set_segments(segments=segments)
        return lines


def plot_car(axis, player_name: PlayerName, state: VehicleState,
             vg: VehicleGeometry, box):
    L = vg.l
    W = vg.w
    car_color = vg.colour
    car: Tuple[Tuple[float, float], ...] = \
        ((-L, -W), (-L, +W), (+L, +W), (+L, -W), (-L, -W))
    xy_theta = (state.x, state.y, state.th)
    q = SE2_from_xytheta(xy_theta)
    x1, y1 = get_transformed_xy(q, car)
    if box is None:
        box, = axis.fill([], [], color=car_color, alpha=0.3, zorder=10)
        x4, y4 = get_transformed_xy(q, ((0, 0),))
        axis.text(x4, y4, player_name, zorder=25,
                  horizontalalignment="center",
                  verticalalignment="center")
    box.set_xy(np.array(list(zip(x1, y1))))
    return box


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Number, Number]]) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.float_(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y
