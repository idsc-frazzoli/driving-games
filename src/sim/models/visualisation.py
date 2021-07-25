from games import PlayerName
from sim.models.vehicle import VehicleState, VehicleGeometry
import numpy as np

def plot_vehicle(axis,
             player_name: PlayerName,
             state: VehicleState,
             vg: VehicleGeometry,
             alpha: float,
             box):
    L = vg.l_half
    W = vg.w_half
    car_color = vg.colour
    car: Tuple[Tuple[float, float], ...] = \
        ((-L, -W), (-L, +W), (+L, +W), (+L, -W), (-L, -W))
    xy_theta = (state.x, state.y, state.theta)
    q = SE2_from_xytheta(xy_theta)
    x1, y1 = get_transformed_xy(q, car)
    if box is None:
        box, = axis.fill([], [], color=car_color, alpha=alpha, zorder=10)
        x4, y4 = get_transformed_xy(q, ((0, 0),))
        axis.text(x4, y4, player_name, zorder=25,
                  horizontalalignment="center",
                  verticalalignment="center")
    box.set_xy(np.array(list(zip(x1, y1))))
    return box


def get_transformed_xy(q: np.array, points: Sequence[Tuple[Number, Number]]) -> Tuple[np.array, np.array]:
    car = tuple((x, y, 1) for x, y in points)
    car = np.float_(car).T
    points = q @ car
    x = points[0, :]
    y = points[1, :]
    return x, y