from decimal import Decimal as D
from typing import Mapping, Optional, Sequence, Tuple, Union, FrozenSet, List

from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager

# from matplotlib import image
from geometry import translation_angle_from_SE2
from matplotlib.patches import Polygon

from dg_commons import PlayerName, Timestamp
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.simulator_animation import adjust_axes_limits, lights_colors_from_lights_cmd
from dg_commons.sim.simulator_visualisation import plot_vehicle, ZOrders
from games import GameVisualization
from . import VehicleTrackDynamics
from .dg_def import DgSimpleParams
from .resources_occupancy import CellIdx
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
        dynamics: Mapping[PlayerName, VehicleTrackDynamics],
        plot_limits: Union[str, Sequence[Sequence[float]]] = "auto",  # fixme alre\ady in params
        *args,
        **kwargs
    ):
        self.params: DgSimpleParams = params
        self.geometries: Mapping[PlayerName, VehicleGeometry] = geometries
        self.dynamics: Mapping[PlayerName, VehicleTrackDynamics] = dynamics
        self.plot_limits = plot_limits
        self.commonroad_renderer: MPRenderer = MPRenderer(*args, **kwargs)
        self.pylab = None
        self._shapely_vis = ShapelyViz()

    @contextmanager
    def plot_arena(self, pylab, ax):
        self.pylab = pylab
        self.commonroad_renderer.ax = ax
        self._shapely_vis.ax = ax
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
        dt: Optional[Timestamp] = None,
        vehicle_poly: Optional[List[Polygon]] = None,
        resources_poly: Optional[List[Polygon]] = None,
        opacity: float = 1.0,
    ):  # todo typing
        """Draw the player at a certain state doing certain commands (if given)"""
        q = self.params.ref_lanes[player_name].lane_pose(float(state.x), 0, 0).center_point
        xy, theta = translation_angle_from_SE2(q.as_SE2())
        velocity = float(state.v)
        global_state = VehicleState(x=xy[0], y=xy[1], theta=theta, vx=velocity, delta=float(0))

        vg = self.geometries[player_name]
        # todo adjust the resource plotting
        if dt is not None:  # not too nice to have this triggering the res visualisation
            dyn = self.dynamics[player_name]
            res: FrozenSet[CellIdx]
            res = dyn.get_shared_resources(state, dt)
            for cell in res:
                poly = dyn.resources_occupancy.get_poly_from_idx(cell)
                self._shapely_vis.add_shape(poly, zorder=ZOrders.ENV_OBSTACLE, color=vg.color, alpha=0.5)
        acc = 0 if commands is None else float(commands.acc)
        lights_colors = lights_colors_from_lights_cmd(state.light, acc, t)
        patches, _ = plot_vehicle(
            ax=self.pylab.gca(),
            player_name=player_name,
            state=global_state,
            lights_colors=lights_colors,
            alpha=0.9,
            vg=vg,
            vehicle_poly=vehicle_poly,
            plot_wheels=True,
            edgecolor="k",
        )
        return patches

    def plot_goal(self, player_name: PlayerName):
        goal_progress = self.params.progress[player_name][1]
        q = self.params.ref_lanes[player_name].lane_pose(float(goal_progress), 0, 0).center_point
        xy, theta = translation_angle_from_SE2(q.as_SE2())
        ax = self.pylab.gca()
        color = self.geometries[player_name].color
        ax.scatter(xy[0], xy[1], s=10, marker="o", alpha=0.5, color=color, edgecolors="k", zorder=ZOrders.ENV_OBSTACLE)

    def hint_graph_node_pos(self, state: VehicleTrackState) -> Tuple[float, float]:
        w = -state.wait * D(0.2)
        return float(state.x), float(state.v + w)
