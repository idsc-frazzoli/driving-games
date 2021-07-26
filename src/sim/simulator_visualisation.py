from abc import ABC, abstractmethod

from matplotlib.axes import Axes
from matplotlib.axis import Axis

from sim.models.vehicle import VehicleState, VehicleGeometry, ModelGeometry

from numbers import Number
from typing import Sequence, Tuple, Generic, Optional, Union

import numpy as np
import os
from decorator import contextmanager
from imageio import imread

from games import PlayerName, X, U, Y
from geometry import SE2_from_xytheta

from sim.simulator import SimContext
from sim.typing import Color
from world.map_loading import map_directory

__all__ = ["SimVisualization"]


class SimVisualisationABC(Generic[X, U, Y], ABC):
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


class SimVisualization(SimVisualisationABC):
    """ Visualization for the trajectory games"""

    def __init__(self, sim_context: SimContext):
        self.sim_context = sim_context

    @contextmanager
    def plot_arena(self, ax: Axes):
        png_path = os.path.join(map_directory, f"{self.sim_context.map_name}.png")
        img = imread(png_path)
        tile_size = self.sim_context.map.tile_size
        H = self.sim_context.map["tilemap"].H
        W = self.sim_context.map["tilemap"].W
        x_size = tile_size * W
        y_size = tile_size * H
        ax.imshow(img, extent=[0, x_size, 0, y_size])
        ax.set_xlim(left=0, right=x_size)
        ax.set_ylim(bottom=0, top=y_size)
        yield

    def plot_player(self,
                    ax: Axes,
                    player_name: PlayerName,
                    state: VehicleState,
                    alpha: float = 0.3,
                    box=None):
        """ Draw the player and his action set at a certain state. """

        vg: VehicleGeometry = self.sim_context.models[player_name].get_geometry()
        box = plot_vehicle(ax=ax,
                           player_name=player_name,
                           state=state,
                           vg=vg,
                           alpha=alpha,
                           box=box)
        return box


def plot_vehicle(ax: Axes,
                 player_name: PlayerName,
                 state: VehicleState,
                 vg: VehicleGeometry,
                 alpha: float,
                 box):
    vehicle_outline: Tuple[Tuple[float, float], ...] = vg.outline
    vehicle_color: Color = vg.color
    q = SE2_from_xytheta((state.x, state.y, state.theta))
    x1, y1 = get_transformed_xy(q, vehicle_outline)
    if box is None:
        box, = ax.fill([], [], color=vehicle_color, alpha=alpha, zorder=10)
        x4, y4 = get_transformed_xy(q, ((0, 0),))
        ax.text(x4, y4, player_name, zorder=25,
                horizontalalignment="center",
                verticalalignment="center")
    box.set_xy(np.array(list(zip(x1, y1))))
    return box


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Number, Number]]) -> Tuple[np.array, np.array]:
    vehicle = tuple((x, y, 1) for x, y in points)
    vehicle = np.float_(vehicle).T
    points = q @ vehicle
    x = points[0, :]
    y = points[1, :]
    return x, y
