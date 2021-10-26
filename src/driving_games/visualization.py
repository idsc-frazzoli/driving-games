from decimal import Decimal as D
from typing import Any, Mapping, Optional, Tuple

from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager

# from matplotlib import image
from geometry import translation_angle_from_SE2

from dg_commons import PlayerName, Timestamp
from dg_commons.maps import DgLanelet
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.simulator_animation import lights_colors_from_lights_cmd
from dg_commons.sim.simulator_visualisation import plot_vehicle
from driving_games.dg_def import DGSimpleParams
from driving_games.structures import VehicleActions, VehicleCosts, VehicleTrackState
from driving_games.vehicle_observation import VehicleObs
from games import GameVisualization

__all__ = ["DrivingGameVisualization"]


class DrivingGameVisualization(
    GameVisualization[VehicleTrackState, VehicleActions, VehicleObs, VehicleCosts, CollisionReportPlayer]
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
            self.commonroad_renderer,
            draw_params={"traffic_light": {"draw_traffic_lights": False}},
        )
        self.commonroad_renderer.render()
        yield
        pylab.axis("off")
        ax.set_aspect("equal")

    def plot_player(
        self,
        player_name: PlayerName,
        state: VehicleTrackState,
        commands: Optional[VehicleActions],
        ref: DgLanelet,
        t: Timestamp,
        opacity: float = 1.0,
    ):
        """Draw the player at a certain state doing certain commands (if given)"""
        q = ref.lane_pose(float(state.x), 0, 0)
        xy, theta = translation_angle_from_SE2(q)
        velocity = float(state.v)
        global_state = VehicleState(x=xy[0], y=xy[1], theta=theta, vx=velocity, delta=float(0))

        vg = self.geometries[player_name]
        # todo adjust the resource plotting
        # resources: FrozenSet[Rectangle]
        # vcolor = np.array(vg.color) * 0.5 + np.array([0.5, 0.5, 0.5]) * 0.5
        # resources = get_resources_used(vs=state, vg=vg, ds=self.ds)
        # for rectangle in resources:
        #     countour_points = np.array(get_rectangle_countour(rectangle)).T
        #
        #     x, y = countour_points[0, :], countour_points[1, :]
        #
        #     self.pylab.plot(x, y, "-", linewidth=0.3, color=vcolor)
        lights_colors = lights_colors_from_lights_cmd(state.light, float(commands.acc), t)
        plot_vehicle(
            ax=self.pylab.gca(),
            player_name=player_name,
            state=global_state,
            lights_colors=lights_colors,
            alpha=0.9,
            vg=vg,
            plot_wheels=True,
        )

    def hint_graph_node_pos(self, state: VehicleTrackState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)
