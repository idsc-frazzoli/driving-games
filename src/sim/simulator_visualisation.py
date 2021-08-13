from abc import ABC, abstractmethod
from enum import IntEnum
from math import inf
from typing import Sequence, Tuple, Generic, Optional, List, Union

import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager
from geometry import SE2_from_xytheta
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from games import PlayerName, X, U, Y
from sim.models.pedestrian import PedestrianState, PedestrianGeometry
from sim.models.vehicle import VehicleState, VehicleGeometry
from sim.simulator import SimContext
from sim.types import Color

__all__ = ["SimRenderer"]


class SimRendererABC(Generic[X, U, Y], ABC):
    """ An artist that can draw the game. """

    @abstractmethod
    def plot_arena(self, ax: Axes):
        """ Context manager. Plots the arena. """
        pass

    @abstractmethod
    def plot_player(
            self,
            ax: Axes,
            player_name: PlayerName,
            state: X,
            alpha: float = 1.0,
            box=None
    ):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        pass


class SimRenderer(SimRendererABC):
    """ Visualization for the trajectory games"""

    def __init__(self, sim_context: SimContext, ax: Axes = None, *args, **kwargs):
        self.sim_context = sim_context
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, **kwargs)

    @contextmanager
    def plot_arena(self, ax: Axes):
        # planning_problem_set.draw(rnd)
        self.sim_context.scenario.lanelet_network.draw(self.commonroad_renderer, draw_params={"traffic_light": {
            "draw_traffic_lights": False}})
        self.commonroad_renderer.render()
        yield

    def plot_player(self,
                    ax: Axes,
                    player_name: PlayerName,
                    state: X,
                    alpha: float = 0.3,
                    polygons: Optional[List[Polygon]] = None,
                    plot_wheels: bool = False) -> List[Polygon]:
        """ Draw the player and his action set at a certain state. """

        mg = self.sim_context.models[player_name].get_geometry()
        if issubclass(type(state), VehicleState):
            polygons = plot_vehicle(ax=ax,
                                    player_name=player_name,
                                    state=state,
                                    vg=mg,
                                    alpha=alpha,
                                    boxes=polygons,
                                    plot_wheels=plot_wheels)
        else:
            polygons = plot_pedestrian(ax=ax,
                                       player_name=player_name,
                                       state=state,
                                       pg=mg,
                                       alpha=alpha,
                                       boxes=polygons,
                                       )
        return polygons


class ZOrders(IntEnum):
    MODEL = 35
    PLAYER_NAME = 40


def plot_vehicle(ax: Axes,
                 player_name: PlayerName,
                 state: VehicleState,
                 vg: VehicleGeometry,
                 alpha: float,
                 boxes: Optional[List[Polygon]],
                 plot_wheels: bool = False) -> List[Polygon]:
    vehicle_outline: Sequence[Tuple[float, float], ...] = vg.outline
    vehicle_color: Color = vg.color
    q = SE2_from_xytheta((state.x, state.y, state.theta))
    if boxes is None:
        vehicle_box = ax.fill([], [], color=vehicle_color, alpha=alpha, zorder=ZOrders.MODEL)[0]
        boxes = [vehicle_box, ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        ax.text(x4, y4, player_name, zorder=ZOrders.PLAYER_NAME,
                horizontalalignment="center",
                verticalalignment="center")
        if plot_wheels:
            wheels_boxes = [ax.fill([], [], color=vehicle_color, alpha=alpha, zorder=ZOrders.MODEL)[0] for _ in
                            range(vg.n_wheels)]
            boxes.extend(wheels_boxes)
    outline = transform_xy(q, vehicle_outline)
    boxes[0].set_xy(outline)
    if plot_wheels:
        wheels_outlines = vg.get_rotated_wheels_outlines(state.delta)
        wheels_outlines = [q @ w_outline for w_outline in wheels_outlines]
        for w_idx, wheel in enumerate(boxes[1:]):
            xy_poly = wheels_outlines[w_idx][:2, :].T
            wheel.set_xy(xy_poly)
    return boxes


def plot_pedestrian(ax: Axes,
                    player_name: PlayerName,
                    state: PedestrianState,
                    pg: PedestrianGeometry,
                    alpha: float,
                    boxes: Optional[List[Polygon]]) -> List[Polygon]:
    q = SE2_from_xytheta((state.x, state.y, state.theta))
    if boxes is None:
        pedestrian_box = ax.fill([], [], color=pg.color, alpha=alpha, zorder=ZOrders.MODEL)[0]
        boxes = [pedestrian_box, ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        ax.text(x4, y4, player_name, zorder=ZOrders.PLAYER_NAME, horizontalalignment="center",
                verticalalignment="center")
    ped_outline: Sequence[Tuple[float, float], ...] = pg.outline
    outline_xy = transform_xy(q, ped_outline)
    boxes[0].set_xy(outline_xy)
    return boxes


def plot_history(ax: Axes,
                 state: VehicleState,
                 vg: VehicleGeometry,
                 traces: Optional[Line2D] = None
                 ):
    if traces is None:
        trace, = ax.plot([], [], ',-', lw=1)
    # todo similar to https://matplotlib.org/stable/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py


def transform_xy(q: np.ndarray, points: Sequence[Tuple[float, float]]) -> Sequence[Tuple[float, float]]:
    points_array = np.array([(x, y, 1) for x, y in points]).T
    points = q @ points_array
    x = points[0, :]
    y = points[1, :]
    return list(zip(x, y))


def approximate_bounding_box_players(obj_list: Sequence[X]) -> Union[Sequence[List], None]:
    minmax = [[inf, -inf], [inf, -inf]]
    for state in obj_list:
        x, y = state.x, state.y
        for i in range(2):
            xory = x if i == 0 else y
            if xory < minmax[i][0]:
                minmax[i][0] = xory
            if xory > minmax[i][1]:
                minmax[i][1] = xory
    if not (max(minmax) == inf and min(minmax) == -inf):
        for i in range(2):
            assert minmax[i][0] <= minmax[i][1]
            minmax[i][0] -= 10
            minmax[i][1] += 10
        return minmax
    return None
