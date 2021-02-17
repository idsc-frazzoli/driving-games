from numbers import Number
from typing import Any, Sequence, Tuple

import numpy as np
import os
from decorator import contextmanager
from duckietown_world import DuckietownMap
from imageio import imread
from matplotlib import patches
from matplotlib.axes import Axes
from networkx import MultiDiGraph, draw_networkx_nodes, draw_networkx_edges

from games import PlayerName
from geometry import SE2_from_xytheta

from world.map_loading import map_directory, load_driving_game_map
from .structures import VehicleActions, VehicleGeometry, VehicleState
from .static_game import GameVisualization, StaticGamePlayer
from .trajectory_world import TrajectoryWorld

__all__ = ["TrajGameVisualization"]

VehicleObservation = None
VehicleCosts = None
Collision = None


class TrajGameVisualization(GameVisualization[VehicleState, VehicleActions, TrajectoryWorld]):
    """ Visualization for the trajectory games"""

    world: TrajectoryWorld
    pylab: Any
    grid: DuckietownMap

    def __init__(self, world: TrajectoryWorld):
        self.world = world
        self.pylab = None
        self.grid = load_driving_game_map(name=world.map_name)

    @contextmanager
    def plot_arena(self, pylab, ax):

        png_path = os.path.join(map_directory, f"{self.world.map_name}.png")
        img = imread(png_path)
        tile_size = self.grid.tile_size
        H = self.grid['tilemap'].H
        W = self.grid['tilemap'].W
        x_size = tile_size * W
        y_size = tile_size * H
        pylab.imshow(img, extent=[0, x_size, 0, y_size])
        ax.set_xlim(left=0, right=x_size)
        ax.set_ylim(bottom=0, top=y_size)
        self.pylab = pylab

        for player, lane in self.world.lanes.items():
            points = lane.lane_profile()
            xp, yp = zip(*points)
            x = np.array(xp)
            y = np.array(yp)
            pylab.fill(x, y, color=self.world.get_geometry(player).colour, alpha=.1, zorder=1)

        yield
        # pylab.grid()

    def plot_player(
            self,
            player_name: PlayerName,
            state: VehicleState,
    ):
        """ Draw the player and his action set at a certain state. """

        vg: VehicleGeometry = self.world.get_geometry(player_name)
        plot_car(
            pylab=self.pylab,
            player_name=player_name,
            state=state,
            vg=vg,
        )

    def plot_actions(self, player: StaticGamePlayer):
        G: MultiDiGraph = player.graph
        colour = player.vg.colour

        def pos_node(n: VehicleState):
            x = G.nodes[n]["x"]
            y = G.nodes[n]["y"]
            return float(x), float(y)

        def line_width(n: VehicleState):
            return float(1.0 / pow(3.0, G.edges[n]["gen"]))

        def node_sizes(n: VehicleState):
            return float(1.0 / pow(4.0, G.nodes[n]["gen"]))

        pos = {_: pos_node(_) for _ in G.nodes}
        widths = [line_width(_) for _ in G.edges]
        node_size = [node_sizes(_) for _ in G.nodes]
        nodes = draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=G.nodes(),
            node_size=node_size,
            node_color='grey',
            alpha=0.5,
        )
        edges = draw_networkx_edges(
            G,
            pos=pos,
            edgelist=G.edges(),
            alpha=0.5,
            arrows=False,
            width=widths,
            edge_color=colour,
        )
        ax: Axes = self.pylab.gca()
        nodes.set_zorder(20)
        edges.set_zorder(5)
        ax.add_collection(nodes)
        ax.add_collection(edges)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


def plot_car(
        pylab,
        player_name: PlayerName,
        state: VehicleState,
        vg: VehicleGeometry,
):
    PLOT_VEL = False
    L = float(vg.l)
    W = float(vg.w)
    car_color = vg.colour
    car: Tuple[Tuple[float, float], ...] = \
        ((-L, -W), (-L, +W), (+L, +W), (+L, -W), (-L, -W))
    xy_theta = tuple(float(_) for _ in (state.x, state.y, state.th))
    q = SE2_from_xytheta(xy_theta)
    x1, y1 = get_transformed_xy(q, car)
    pylab.fill(x1, y1, color=car_color, alpha=.3, zorder=10)

    v_size = float(state.v) * 0.2
    v_vect = ((+L, 0), (+L + v_size, 0))
    x3, y3 = get_transformed_xy(q, v_vect)
    arrow = patches.Arrow(x=x3[0], y=y3[0], dx=x3[1] - x3[0], dy=y3[1] - y3[0], facecolor=car_color, edgecolor='k')
    if PLOT_VEL:
        pylab.gca().add_patch(arrow)

    x4, y4 = get_transformed_xy(q, ((0, 0),))
    pylab.text(
        x4,
        y4,
        player_name,
        zorder=25,
        horizontalalignment="center",
        verticalalignment="center",
    )


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Number, Number]]) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.float_(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y
