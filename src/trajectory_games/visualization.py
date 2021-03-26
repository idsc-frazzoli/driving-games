from numbers import Number
from typing import Sequence, Tuple, Mapping, FrozenSet

import numpy as np
import os
from decorator import contextmanager
from duckietown_world import DuckietownMap
from imageio import imread
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels

from games import PlayerName
from geometry import SE2_from_xytheta

from world.map_loading import map_directory, load_driving_game_map
from .structures import VehicleGeometry, VehicleState
from .sequence import Timestamp
from .paths import Trajectory, Transition
from .static_game import GameVisualization, StaticGamePlayer
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
    def plot_arena(self, pylab, ax):

        png_path = os.path.join(map_directory, f"{self.world.map_name}.png")
        img = imread(png_path)
        tile_size = self.grid.tile_size
        H = self.grid["tilemap"].H
        W = self.grid["tilemap"].W
        x_size = tile_size * W
        y_size = tile_size * H
        pylab.imshow(img, extent=[0, x_size, 0, y_size])
        ax.set_xlim(left=0, right=x_size)
        ax.set_ylim(bottom=0, top=y_size)

        for player, lane in self.world.lanes.items():
            points = lane.lane_profile()
            xp, yp = zip(*points)
            x = np.array(xp)
            y = np.array(yp)
            pylab.fill(x, y, color=self.world.get_geometry(player).colour, alpha=0.1, zorder=1)

        yield
        # pylab.grid()

    def plot_player(self, pylab, player_name: PlayerName,
                    state: VehicleState, box=None):
        """ Draw the player and his action set at a certain state. """

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        box = plot_car(pylab=pylab, player_name=player_name,
                       state=state, vg=vg, box=box)
        return box

    def plot_actions(self, pylab, player: StaticGamePlayer):
        ax: Axes = pylab.gca()

        state = next(iter(player.state.support()))
        trajectories = player.actions_generator.get_action_set(state=state, world=None, player=player.name)
        self.plot_trajectories(pylab=pylab, trajectories=trajectories,
                               colour=player.vg.colour, width=0.5)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    def plot_equilibria(self, pylab, path: Trajectory, player: StaticGamePlayer):

        self.plot_trajectories(pylab=pylab, trajectories=frozenset([path]),
                               colour=player.vg.colour, width=1.0)

        def get_vals(trans: Transition) -> Tuple[float, float, Timestamp]:
            xs = trans.states[1]
            return xs.x, xs.y, xs.t
        vals = [get_vals(trans=transition) for transition in path]
        x, y, t = zip(*vals)
        pylab.scatter(x, y, s=10.0, c=t, zorder=10)

    def plot_pref(self, pylab, player: StaticGamePlayer,
                  origin: Tuple[float, float],
                  labels: Mapping[WeightedPreference, str] = None):

        assert isinstance(player.preference, PosetalPreference),\
            f"Preference is of type {player.preference.get_type()}" \
            f" and not {PosetalPreference.get_type()}"
        X, Y = origin
        G: DiGraph = player.preference.graph

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
        draw_networkx_edges(
            G,
            pos=pos,
            edgelist=G.edges(),
            arrows=True,
            arrowstyle="-",
        )

        ax: Axes = pylab.gca()
        draw_networkx_labels(G, pos=pos, labels=labels, ax=ax, font_size=8, font_color="b")
        ax.text(x=X, y=Y+10.0, s=player.name+text, ha="center", va="center")

    @staticmethod
    def plot_trajectories(pylab, trajectories: FrozenSet[Trajectory],
                          colour: Tuple[float, float, float], width: float):
        segments = []
        for traj in trajectories:
            sampled_traj = np.array([np.array([v.x, v.y])
                                     for v in traj.get_sampled_trajectory()[1]])
            segments.append(sampled_traj)

        lines = LineCollection(segments=segments, colors=colour, linewidths=width)
        lines.set_zorder(5)
        ax: Axes = pylab.gca()
        ax.add_collection(lines)


def plot_car(pylab, player_name: PlayerName, state: VehicleState,
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
        box, = pylab.fill([], [], color=car_color, alpha=0.3, zorder=10)
        x4, y4 = get_transformed_xy(q, ((0, 0),))
        pylab.text(x4, y4, player_name, zorder=25,
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
