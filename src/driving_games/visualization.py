from decimal import Decimal as D
from typing import Mapping, Optional, Sequence, Tuple, Union

from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager

from dg_commons import PlayerName, Timestamp
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.simulator_animation import adjust_axes_limits, lights_colors_from_lights_cmd
from dg_commons.sim.simulator_visualisation import plot_vehicle, ZOrders
from games import GameVisualization
# from matplotlib import image
from geometry import translation_angle_from_SE2
from .dg_def import DgSimpleParams
from .structures import VehicleActions, VehicleTimeCost, VehicleTrackState
from .vehicle_observation import VehicleObs

__all__ = ["DrivingGameVisualization"]


class DrivingGameVisualization(
    GameVisualization[VehicleTrackState, VehicleActions, VehicleObs, VehicleTimeCost, CollisionReportPlayer]
):
    """Visualization for the driving games"""

    def __init__(
        self,
        params: DgSimpleParams,
        geometries: Mapping[PlayerName, VehicleGeometry],
        ds: D,
        plot_limits: Union[str, Sequence[Sequence[float]]] = "auto",
        *args,
        **kwargs
    ):
        self.params: DgSimpleParams = params
        self.commonroad_renderer: MPRenderer = MPRenderer(*args, **kwargs)
        self.geometries: Mapping[PlayerName, VehicleGeometry] = geometries
        self.ds: D = ds
        self.plot_limits = plot_limits
        self.pylab = None

    @contextmanager
    def plot_arena(self, pylab, ax):
        self.pylab = pylab
        self.commonroad_renderer.ax = ax
        self.params.scenario.lanelet_network.draw(
            self.commonroad_renderer,
            draw_params={"traffic_light": {"draw_traffic_lights": False}},
        )
        self.commonroad_renderer.render()
        # plot goals
        for pn in self.params.progress:
            self.plot_goal(pn)

        yield
        # pylab.axis("off")
        adjust_axes_limits(ax=ax, plot_limits=self.plot_limits)
        ax.set_aspect("equal")

    def plot_player(
        self,
        player_name: PlayerName,
        state: VehicleTrackState,
        commands: Optional[VehicleActions],
        t: Timestamp,
        opacity: float = 1.0,
    ):
        """Draw the player at a certain state doing certain commands (if given)"""
        q = self.params.ref_lanes[player_name].lane_pose(float(state.x), 0, 0).center_point
        xy, theta = translation_angle_from_SE2(q.as_SE2())
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
        acc = 0 if commands is None else float(commands.acc)
        lights_colors = lights_colors_from_lights_cmd(state.light, acc, t)
        plot_vehicle(
            ax=self.pylab.gca(),
            player_name=player_name,
            state=global_state,
            lights_colors=lights_colors,
            alpha=0.9,
            vg=vg,
            plot_wheels=True,
        )

    def plot_goal(self, player_name: PlayerName):
        goal_progress = self.params.progress[player_name][1]
        q = self.params.ref_lanes[player_name].lane_pose(float(goal_progress), 0, 0).center_point
        xy, theta = translation_angle_from_SE2(q.as_SE2())
        ax = self.pylab.gca()
        color = self.geometries[player_name].color
        ax.plot(xy[0], xy[1], "o", markersize=5, alpha=0.5, color=color, zorder=ZOrders.ENV_OBSTACLE)

    def hint_graph_node_pos(self, state: VehicleTrackState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)
