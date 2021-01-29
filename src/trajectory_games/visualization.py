from numbers import Number
from typing import Any, Sequence, Tuple, Dict

import numpy as np
from decorator import contextmanager
from matplotlib import patches
from matplotlib.axes import Axes
from networkx import MultiDiGraph, draw_networkx_nodes, draw_networkx_edges

from games import PlayerName
from geometry import SE2_from_xytheta
from .structures import VehicleActions, VehicleGeometry, VehicleState
from .static_game import GameVisualization, StaticGamePlayer
from .world import World
from .paths import PathWithBounds

__all__ = ["TrajGameVisualization"]

VehicleObservation = None
VehicleCosts = None
Collision = None


class TrajGameVisualization(GameVisualization[VehicleState, VehicleActions, World]):
    """ Visualization for the trajectory games"""

    world: World
    pylab: Any

    def __init__(self, world: World):
        self.world = world
        self.pylab = None

    @contextmanager
    def plot_arena(self, pylab, ax):

        side: float = 0.1  # Additional space on sides (scale of length)
        disc: float = 1.0  # discretisation (m)

        x_max, x_min, y_max, y_min = -1000., 1000., -1000., 1000.
        paths: Dict[PlayerName, PathWithBounds] = {}
        path_patches = {}
        for player in self.world.get_players():
            path = self.world.get_reference(player=player)
            s_min, s_max = path.get_s_limits()
            s_min, s_max = float(s_min), float(s_max)
            ds = int(s_max - s_min // disc)
            s = np.linspace(s_min, s_max, ds)
            n_min, n_max = zip(*path.get_bounds_at_s(s))
            sn_min = list(zip(s, n_min))
            sn_max = list(zip(s, n_max))
            xy_min = np.float_(path.curvilinear_to_cartesian(sn_min))
            xy_max = np.float_(path.curvilinear_to_cartesian(sn_max))
            poly_points = np.vstack([xy_min, np.flipud(xy_max)])
            x_min = min(x_min, np.min(poly_points[:, 0]))
            x_max = max(x_max, np.max(poly_points[:, 0]))
            y_min = min(y_min, np.min(poly_points[:, 1]))
            y_max = max(y_max, np.max(poly_points[:, 1]))
            # colour = self.world.get_geometry(player).colour
            path_patches[player] = patches.Polygon(poly_points, linewidth=0, edgecolor="r",
                                                   facecolor="lightgray")
            paths[player] = path

        points = ((x_min, y_min), (x_max, y_min), (x_max, y_max),
                  (x_min, y_max), (x_min, y_min))
        px, py = zip(*points)
        pylab.plot(px, py, "k-")
        self.pylab = pylab

        x_lim, y_lim = x_max - x_min, y_max - y_min
        side *= max(x_lim, y_lim)
        grass = patches.Rectangle((x_min - side, y_min - side),
                                  x_lim + 2 * side, y_lim + 2 * side,
                                  linewidth=0, edgecolor="r", facecolor="green")
        ax.add_patch(grass)
        for _, patch in path_patches.items():
            ax.add_patch(patch)
            pass

        yield
        pylab.axis((x_min - 2 * side, x_max + 2 * side, y_min - 2 * side, y_max + 2 * side))
        # pylab.axis("off")
        # pylab.xlabel("x")
        # pylab.ylabel("y")
        ax.set_aspect("equal")

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
