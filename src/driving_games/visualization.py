from decimal import Decimal as D
from numbers import Number
from typing import Optional, Sequence, Tuple

import numpy as np
from decorator import contextmanager
from matplotlib import patches

from games import GameVisualization, PlayerName
from .driving_example import SE2_from_VehicleState
from .structures import CollisionCost, VehicleActions, VehicleObservation, VehicleState


class DrivingGameVisualization(
    GameVisualization[VehicleState, VehicleActions, VehicleObservation, D, CollisionCost]
):
    side: D

    def __init__(self, params, side: D):
        self.params = params
        self.side = side

    @contextmanager
    def plot_arena(self, pylab, ax):
        params = self.params
        side = float(params.side)
        road = float(params.road)
        L = float(params.side + params.road + params.side)
        start = params.side + params.road_lane_offset
        points = ((0, 0), (L, 0), (L, L), (0, L), (0, 0))
        px, py = zip(*points)
        # logger.info(px=px, py=py, points=points)
        pylab.plot(px, py, "k-")
        self.pylab = pylab

        # t2 = pylab.transforms.Affine2D().rotate_deg(-45) + ax.transData
        # r2.set_transform(t2)

        grass = patches.Rectangle((0, 0), L, L, linewidth=0, edgecolor="r", facecolor="green")
        ax.add_patch(grass)
        # Create a Rectangle patch
        rect = patches.Rectangle((side, 0), road, L, linewidth=0, edgecolor="r", facecolor="grey")
        # Add the patch to the Axes
        ax.add_patch(rect)
        rect = patches.Rectangle((0, side), L, road, linewidth=0, edgecolor="r", facecolor="grey")
        # Add the patch to the Axes
        ax.add_patch(rect)

        yield
        b = 0.1 * L
        pylab.axis((0 - b, L + b, 0 - b, L + b))
        pylab.axis("off")
        ax.set_aspect("equal")

    def plot_player(
        self,
        player_name: PlayerName,
        state: VehicleState,
        commands: Optional[VehicleActions],
        opacity: float = 1.0,
    ):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        q = SE2_from_VehicleState(state)

        if commands is None:
            light = "none"
        else:
            light = commands.light

        colors = {
            "none": {
                "back_left": "red",
                "back_right": "red",
                "front_right": "white",
                "front_left": "white",
            },
            # "headlights", "turn_left", "turn_right"
        }
        velocity = float(state.v)
        plot_car(self.pylab, q, velocity=velocity, car_color="blue", light_colors=colors[light])

    def hint_graph_node_pos(self, state: VehicleState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)


def plot_car(pylab, q: np.array, velocity, car_color, light_colors):
    L: float = 4.0
    W: float = 2.5
    car: Tuple[Tuple[float, float], ...] = (
        (-L / 2, -W / 2),
        (-L / 2, +W / 2),
        (+L / 2, +W / 2),
        (+L / 2, -W / 2),
        (-L / 2, -W / 2),
    )
    x1, y1 = get_transformed_xy(q, car)
    pylab.fill(x1, y1, color=car_color, zorder=10)

    l: float = 0.1 * L
    radius_light = 0.03 * L
    light_position = {
        "back_left": (-L / 2, +W / 2 - l),
        "back_right": (-L / 2, -W / 2 + l),
        "front_left": (+L / 2, +W / 2 - l),
        "front_right": (+L / 2, -W / 2 + l),
    }
    for name in light_position:
        light_color = light_colors[name]
        position = light_position[name]
        x2, y2 = get_transformed_xy(q, (position,))
        patch = patches.Circle((x2[0], y2[0]), radius=radius_light, color=light_color)
        ax = pylab.gca()
        ax.add_patch(patch)

    arrow = ((+L / 2, 0), (+L / 2 + velocity, 0))
    x3, y3 = get_transformed_xy(q, arrow)
    pylab.plot(x3, y3, "r-", zorder=99)

    x4, y4 = get_transformed_xy(q, ((0, 0),))
    pylab.plot(x4, y4, "k*", zorder=15)


def get_transformed_xy(
    q: np.array, points: Sequence[Tuple[Number, Number]]
) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.array(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y
