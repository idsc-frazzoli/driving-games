import os
from decimal import Decimal as D
from numbers import Number
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple
from cairosvg import svg2png

import numpy as np
from decorator import contextmanager
from matplotlib import patches
from matplotlib.image import imread

from duckietown_world.svg_drawing.misc import draw_static

from games import PlayerName, GameVisualization

from driving_games.collisions_check import Collision
from driving_games.structures import SE2_disc

from world.utils import from_SE2_disc_to_SE2
from world.map_loading import map_directory

from duckie_games.duckie_observations import DuckieObservation
from duckie_games.structures import (
    DuckieGeometry,
    DuckieActions,
    DuckieState,
    DuckieCosts
)
from duckie_games.rectangle import Rectangle
from duckie_games.shared_resources import DrivingGameGridMap, ResourceID


class DuckieGameVisualization(GameVisualization[DuckieState, DuckieActions, DuckieObservation, DuckieCosts, Collision]):
    """ Visualization for the duckie games"""

    duckie_map: DrivingGameGridMap
    map_name: str
    side: D
    geometries: Mapping[PlayerName, DuckieGeometry]
    ds: D
    dt: D
    pylab: Any

    def __init__(
            self,
            duckie_map: DrivingGameGridMap,
            map_name: str,
            geometries: Mapping[PlayerName, DuckieGeometry],
            ds: D,
            dt: D
    ):
        self.duckie_map = duckie_map
        self.map_name = map_name
        self.ds = ds
        self.dt = dt
        self.geometries = geometries
        self.pylab = None

    @contextmanager
    def plot_arena(self, pylab, ax):
        d = "out"
        m = self.duckie_map

        png_path = os.path.join(map_directory, f"{self.map_name}.png")

        try:
            img = imread(png_path)
        except (FileNotFoundError, SyntaxError):  # Catch Syntax error in circle ci because PNGs are stored with git lfs
            outdir = os.path.join(d, "map_drawing", f"{self.map_name}")
            svg_path = os.path.join(outdir, "drawing.svg")
            # todo find converter without the render bug (or display html directly)
            if not os.path.exists(svg_path):
                draw_static(m, outdir)
                svg2png(url=svg_path, write_to=png_path)
            img = imread(png_path)

        # logger.info(px=px, py=py, points=points)
        tile_size = m.tile_size
        H = m['tilemap'].H
        W = m['tilemap'].W
        x_size = tile_size * W
        y_size = tile_size * H
        pylab.imshow(img, extent=[0, x_size, 0, y_size])
        ax.set_xlim(left=0, right=x_size)
        ax.set_ylim(bottom=0, top=y_size)
        self.pylab = pylab

        yield
        # b = 0.1 * L
        # pylab.axis((0 - b, L + b, 0 - b, L + b))
        pylab.axis("off")
        ax.set_aspect("equal")

    def plot_player(
            self,
            player_name: PlayerName,
            state: DuckieState,
            commands: Optional[DuckieActions],
            opacity: float = 1.0,
    ):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        q_SE2disc: SE2_disc = state.abs_pose
        q = from_SE2_disc_to_SE2(q_SE2disc)

        if commands is None:
            light = "none"
        else:
            light = commands.light

        # TODO: finish here
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
        vg = self.geometries[player_name]
        resources: FrozenSet[ResourceID]
        vcolor = np.array(vg.color) * 0.5 + np.array([0.5, 0.5, 0.5]) * 0.5
        resources = self.duckie_map.get_resources_used(vs=state, vg=vg, dt=self.dt)

        for _id in resources:
            center_x, center_y = self.duckie_map.resources[_id]
            rec = Rectangle(
                center_pose=(center_x, center_y, D(0)),
                width=self.ds,
                height=self.ds
            )
            countour_points = np.array(rec.closed_contour).T
            x, y = countour_points[0, :], countour_points[1, :]
            self.pylab.fill(x, y, linewidth=0.05, color=vcolor, alpha=0.6, zorder=15)

        plot_car(
            self.pylab,
            player_name,
            q,
            velocity=velocity,
            light_colors=colors[light],
            vg=vg,
        )

    def hint_graph_node_pos(self, state: DuckieState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)


def plot_car(
        pylab,
        player_name: PlayerName,
        q: np.array,
        velocity,
        light_colors,
        vg: DuckieGeometry,
):
    L = float(vg.length)
    W = float(vg.width)
    car_color = vg.color
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
    pylab.plot(x3, y3, "r-", zorder=20)

    x4, y4 = get_transformed_xy(q, ((0, 0),))
    # pylab.plot(x4, y4, "k*", zorder=15)
    pylab.text(
        x4,
        y4,
        player_name,
        zorder=30,
        horizontalalignment="center",
        verticalalignment="center",
    )


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Number, Number]]) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.array(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y
