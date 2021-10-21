from dataclasses import dataclass
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import PlayerObservations
from typing import MutableMapping, Dict, Optional, Tuple
from shapely.geometry import Polygon
from dg_commons.geo import SE2_apply_T2, T2value
from dg_commons import X, U
from scipy.integrate import solve_ivp
import math
from dg_commons.sim.models.vehicle import VehicleParameters, VehicleGeometry
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import PatchCollection


@dataclass
class SituationObservations:
    my_name: PlayerName

    agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None

    rel_poses: Optional[Dict[PlayerName, SE2Transform]] = None

    dt_commands: Optional[float] = None


def relative_velocity(my_vel: float, other_vel: float, transform):
    other_vel_wrt_other = [float(other_vel), 0.0]
    other_vel_wrt_myself = SE2_apply_T2(transform, other_vel_wrt_other)
    return my_vel - other_vel_wrt_myself[0]


def l_w_from_rectangle(occupacy: Polygon):
    x, y = occupacy.exterior.xy[0], occupacy.exterior.xy[1]
    length = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[3], y[3]]))
    width = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
    return length, width


def occupancy_prediction(current_state: X, time_span: float,
                         occupacy: Polygon, vehicle_geom=VehicleGeometry.default_car()):
    dt = 0.2
    safety_factor = 0.5  # metres to keep
    n, rest = int(time_span/dt), time_span % dt
    dts = [dt for _ in range(n)] + [rest]
    l_polygon, r_polygon = [], []
    lr = vehicle_geom.lr

    length, width = l_w_from_rectangle(occupacy)
    length, width = length + safety_factor, width + safety_factor

    def polygon_data(res: np.ndarray):
        theta, pos = res[2], res[:2]
        vec = np.array([-math.sin(theta), math.cos(theta)])*width/2
        l_polygon.append(tuple(pos + vec))
        r_polygon.insert(0, tuple(pos - vec))

    delta, vx = current_state.delta, current_state.vx

    def _dynamics(t: float, state: np.ndarray) -> np.ndarray:
        x_dot = vx * math.cos(state[2])
        y_dot = vx * math.sin(state[2])
        theta_dot = vx * math.tan(delta) / length

        return np.array([x_dot, y_dot, theta_dot])

    initial_position = np.array([current_state.x, current_state.y]) - \
                       np.array([math.cos(current_state.theta), math.sin(current_state.theta)]) * lr
    initial_position_poly = np.array([current_state.x, current_state.y]) - \
                            np.array([math.cos(current_state.theta), math.sin(current_state.theta)]) * length/2
    initial_state = np.array([initial_position[0], initial_position[1], current_state.theta])
    polygon_data(np.array([initial_position_poly[0], initial_position_poly[1], current_state.theta]))
    for dt in dts:
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(dt)), y0=initial_state)
        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")

        res = result.y[:, -1]
        initial_state = res
        polygon_data(res)

    final_state = np.array([res[0] + math.cos(res[2]) * length,
                            res[1] + math.sin(res[2]) * length, res[2]])
    polygon_data(final_state)
    polygon = Polygon(tuple(l_polygon + r_polygon + [l_polygon[0]]))
    return polygon


def entry_exit_t(intersection: Polygon, current_state, occupacy: Polygon, safety_t, vel,
                 vehicle_geom=VehicleGeometry.default_car()):
    distance = occupacy.distance(intersection)
    stopped_inside = vel <= 10e-6 and distance <= 10e-6
    going = vel > 10e-6 and distance > 10e-6
    assert going or stopped_inside

    if stopped_inside:
        return 0, safety_t

    tol = 0.1
    length = intersection.length

    min_t = distance/vel
    max_t = min(min_t + 2 * length / vel, safety_t)

    test_values = list(np.round(np.arange(min_t, max_t, tol), 3))

    entry_t = None
    exit_t = None
    for i, test_value in enumerate(test_values):
        pred = occupancy_prediction(current_state, test_value, occupacy, vehicle_geom=vehicle_geom)
        inter = pred.intersection(intersection)
        if not inter.is_empty:
            if entry_t is None:
                entry_t = test_value - tol/2 if i > 0 else 0
        elif entry_t is not None and exit_t is None:
            exit_t = test_value - tol/2
            break

    if exit_t is None:
        exit_t = safety_t

    return entry_t, exit_t


class PolygonPlotter:
    @dataclass
    class PolygonClass:
        car: bool = False
        dangerous_zone: bool = False
        conflict_area: bool = False

        def __post_init__(self):
            assert self.car or self.dangerous_zone or self.conflict_area

        def get_color(self):
            if self.car:
                return 'b'
            elif self.dangerous_zone:
                return 'lightgray'
            elif self.conflict_area:
                return 'r'

        def get_zorder(self):
            if self.car:
                return 2
            elif self.dangerous_zone:
                return 3
            elif self.conflict_area:
                return 1

    def __init__(self, plot: bool):
        self.counter = 0
        self.plot = plot
        self.frames = {"Frame": [], "Class": []}
        self.current_frame = [[], []]
        self.current_class = []
        self.min_max_x = [math.inf, -math.inf]
        self.min_max_y = [math.inf, -math.inf]
        self.max_n_items = 0
        self.previous_coll = None

    def plot_polygon(self, p: Polygon,  polygon_class: PolygonClass):
        if p.is_empty or not self.plot:
            return

        x, y = list(p.exterior.coords.xy[0][:]), list(p.exterior.coords.xy[1][:])
        min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
        self.min_max_x[0] = min_x if min_x < self.min_max_x[0] else self.min_max_x[0]
        self.min_max_x[1] = max_x if max_x > self.min_max_x[1] else self.min_max_x[1]
        self.min_max_y[0] = min_y if min_y < self.min_max_y[0] else self.min_max_y[0]
        self.min_max_y[1] = max_y if max_y > self.min_max_y[1] else self.min_max_y[1]
        self.current_frame[0].append(x)
        self.current_frame[1].append(y)
        self.current_class.append(polygon_class)

    def next_frame(self):
        n_items = len(self.current_frame[0])
        self.frames["Frame"].append(self.current_frame)
        self.frames["Class"].append(self.current_class)
        self.max_n_items = n_items if n_items > self.max_n_items else self.max_n_items
        self.current_frame = [[], []]

    def save_animation(self, title=""):
        if not self.plot:
            return

        n_frames, enlargement_factor = len(self.frames["Frame"]), 3
        fig = plt.figure()
        self.min_max_x = [self.min_max_x[0] - enlargement_factor, self.min_max_x[1] + enlargement_factor]
        self.min_max_y = [self.min_max_y[0] - enlargement_factor, self.min_max_y[1] + enlargement_factor]
        ax = plt.axes(xlim=tuple(self.min_max_x), ylim=tuple(self.min_max_y))
        plt.gca().set_aspect('equal', adjustable='box')

        polygons = []
        for _ in range(self.max_n_items):
            polygon = patches.Polygon(np.array([[0, 0]]), animated=True)
            polygons.append(polygon)

        def init():
            for polygon in polygons:
                polygon.set_xy(np.array([[0, 0]]))
            return polygons

        def animate(i):
            frame = self.frames["Frame"][i]
            classes = self.frames["Class"][i]
            if self.previous_coll is not None:
                self.previous_coll.remove()

            assert len(frame[0]) == len(frame[1])
            n = len(frame[0])
            for i in range(self.max_n_items):
                if i < n:
                    x = np.array([frame[0][i]]).T
                    y = np.array([frame[1][i]]).T
                    xy = np.concatenate((x, y), axis=1)
                    polygons[i].set_xy(xy)
                    polygons[i].set_zorder(classes[i].get_zorder())
                    polygons[i].set_color(classes[i].get_color())
                else:
                    polygons[i].set_xy(np.array([[0, 0]]))

            p = PatchCollection(polygons, alpha=0.4, match_original=True)
            ax.add_collection(p)
            self.previous_coll = p

            return polygons

        Writer = animation.writers['pillow']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=20, blit=True)

        dir = "out_emergency"
        anim.save(os.path.join(dir, 'emergency.gif'), writer=writer)
