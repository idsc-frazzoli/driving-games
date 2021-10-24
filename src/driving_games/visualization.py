# import glob
from decimal import Decimal as D
from typing import Any, FrozenSet, Mapping, Optional, Tuple

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager
from matplotlib import patches

# from matplotlib import image
from dg_commons import PlayerName
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle_ligths import LightsColors
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.simulator_animation import get_lights_colors_from_cmds
from dg_commons.sim.simulator_visualisation import transform_xy, plot_vehicle
from games import GameVisualization
from . import DGSimpleParams
from .structures import VehicleActions, VehicleCosts, VehicleState
from .vehicle_observation import VehicleObs

__all__ = ["DrivingGameVisualization"]


class DrivingGameVisualization(
    GameVisualization[VehicleState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer]
):
    """Visualization for the driving games"""

    scenario: Scenario
    geometries: Mapping[PlayerName, VehicleGeometry]
    ds: D
    pylab: Any

    def __init__(
        self, params: DGSimpleParams, geometries: Mapping[PlayerName, VehicleGeometry], ds: D, *args, **kwargs
    ):
        self.params = params
        self.commonroad_renderer: MPRenderer = MPRenderer(*args, **kwargs)
        self.ds = ds
        self.pylab = None

    @contextmanager
    def plot_arena(self, pylab, ax):
        self.commonroad_renderer.ax = ax
        self.scenario.lanelet_network.draw(
            self.commonroad_renderer, draw_params={"traffic_light": {"draw_traffic_lights": False}}
        )
        self.commonroad_renderer.render()
        yield
        pylab.axis("off")
        ax.set_aspect("equal")

    def plot_player(
        self,
        player_name: PlayerName,
        state: VehicleState,
        commands: Optional[VehicleActions],
        opacity: float = 1.0,
    ):
        """Draw the player at a certain state doing certain commands (if givne)"""
        # todo fixme
        q = SE2_from_VehicleState(state)

        lights_colors: LightsColors = get_lights_colors_from_cmds(state.light, t=0)
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
        resources: FrozenSet[Rectangle]
        vcolor = np.array(vg.color) * 0.5 + np.array([0.5, 0.5, 0.5]) * 0.5
        resources = get_resources_used(vs=state, vg=vg, ds=self.ds)
        for rectangle in resources:
            countour_points = np.array(get_rectangle_countour(rectangle)).T

            x, y = countour_points[0, :], countour_points[1, :]

            self.pylab.plot(x, y, "-", linewidth=0.3, color=vcolor)

        plot_vehicle(
            self.pylab.gca(),
            player_name,
            q,
            velocity=velocity,
            light_colors=colors[light],
            vg=vg,
        )

    def hint_graph_node_pos(self, state: VehicleState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)


def plot_car(
    pylab,
    player_name: PlayerName,
    q: np.array,
    velocity,
    light_colors,
    vg: VehicleGeometry,
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
    x1, y1 = transform_xy(q, car)
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
        x2, y2 = transform_xy(q, (position,))
        patch = patches.Circle((x2[0], y2[0]), radius=radius_light, color=light_color)
        ax = pylab.gca()
        ax.add_patch(patch)

    arrow = ((+L / 2, 0), (+L / 2 + velocity, 0))
    x3, y3 = transform_xy(q, arrow)
    pylab.plot(x3, y3, "r-", zorder=99)

    x4, y4 = transform_xy(q, ((0, 0),))
    # pylab.plot(x4, y4, "k*", zorder=15)
    pylab.text(
        x4,
        y4,
        player_name,
        zorder=15,
        horizontalalignment="center",
        verticalalignment="center",
    )
