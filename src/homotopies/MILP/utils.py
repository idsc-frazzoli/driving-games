import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()


def state2traj(state: VehicleState, horizon: float, n: int) -> np.ndarray:
    """
    This function takes the current state of an object as input and outputs the predicted trajectory
    assuming constant vx and delta(no control inputs)
    input:
    -state: VehicleState(x,y,theta,vx,delta)
    -horizon: float, duration of the predicted trajectory
    -n: int, number of control points used to represent the trajectory
    output:
    -ctrp: 2*n_ctrp matrix, containing (x,y) coords of the control points
    """
    traj = np.zeros([3, n])
    traj[:, 0] = np.array([state.x,
                           state.y,
                           state.theta])
    t_step = horizon / n
    v = state.vx
    delta = state.delta
    for i in range(n - 1):
        theta = traj[2, i]
        dtheta = v * np.tan(delta) / vehicle_geometry.length
        vy = dtheta * vehicle_geometry.lr
        traj[0, i + 1] = traj[0, i] + t_step * (v * np.cos(theta) - vy * np.sin(theta))
        traj[1, i + 1] = traj[1, i] + t_step * (v * np.sin(theta) + vy * np.cos(theta))
        traj[2, i + 1] = traj[2, i] + t_step * dtheta

    return traj[0:2, :]


def linearize_spline(path: np.ndarray):
    tck, _ = interpolate.splprep(path, k=2)
    u1 = np.linspace(0, 1, 110, endpoint=True)  # don't work for n=~100
    center_x, center_y = interpolate.splev(u1, tck)
    return np.vstack((center_x, center_y))


def find_intersect(path1: np.ndarray, path2: np.ndarray):
    len1 = path1.shape[1]
    len2 = path2.shape[1]
    if len1 < 3 and len2 < 3:
        return check_intersect(path1, path2)
    else:
        if len1 < 3:
            path21 = path2[:, 0:int(len2 / 2) + 1]
            path22 = path2[:, int(len2 / 2):]
            res1, p1 = find_intersect(path1, path21)
            res2, p2 = find_intersect(path1, path22)
            res3 = False
            res4 = False
        elif len2 < 3:
            path11 = path1[:, 0:int(len1 / 2) + 1]
            path12 = path1[:, int(len1 / 2):]
            res1, p1 = find_intersect(path11, path2)
            res2, p2 = find_intersect(path12, path2)
            res3 = False
            res4 = False
        else:
            path11 = path1[:, 0:int(len1 / 2) + 1]
            path12 = path1[:, int(len1 / 2):]
            path21 = path2[:, 0:int(len2 / 2) + 1]
            path22 = path2[:, int(len2 / 2):]
            res1, p1 = find_intersect(path11, path21)
            res2, p2 = find_intersect(path11, path22)
            res3, p3 = find_intersect(path11, path21)
            res4, p4 = find_intersect(path12, path22)
        if res1:
            return res1, p1
        elif res2:
            return res2, p2
        elif res3:
            return res3, p3
        elif res4:
            return res4, p4
        else:
            return False, None


def check_intersect(line1: np.ndarray, line2: np.ndarray):
    """
    this function checks if the two input line segments intersect
    input:
    line1: 2*2 array, [[x1,x2], [y1, y2]]
    line2: 2*2 array, [[x3,x4], [y3, y4]]
    output:
    intersect: bool
    point: 2*1 array
    """
    A1 = line1[1, 1] - line1[1, 0]
    B1 = line1[0, 0] - line1[0, 1]
    C1 = A1 * line1[0, 0] + B1 * line1[1, 0]
    A2 = line2[1, 1] - line2[1, 0]
    B2 = line2[0, 0] - line2[0, 1]
    C2 = A2 * line2[0, 0] + B2 * line2[1, 0]
    det = A1 * B2 - A2 * B1
    if det == 0:  # two lines are parallel
        return False, None
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
    if max(line1[0, :]) >= x >= min(line1[0, :]) and max(line1[1, :]) >= y >= min(line1[1, :])\
            and max(line2[0, :]) >= x >= min(line2[0, :]) and max(line2[1, :]) >= y >= min(line2[1, :]):
        return True, np.array([x, y])
    else:
        return False, None  # intersection point not on segment1


def spline_s(path: np.ndarray, intersect: np.ndarray) -> float:
    """
    this function takes a trajectory(characterized by a sequence of control points) and an intersection point as input
    output the s coord of the intersection point along the path
    input:
    -path: a 2*n array of control points on the trajectory
    -intersect: a 2*1 array of the x-y coords of the intersection point
    output:
    -s: arclength from the start point to the intersection point
    """
    tck, _ = interpolate.splprep(path, k=2)
    p = spline_progress(path, intersect)
    u = np.linspace(0, p, 1000, endpoint=True)
    center_x, center_y = interpolate.splev(u, tck)
    d_center_x = np.insert(np.diff(center_x, axis=0), 0, 0)
    d_center_y = np.insert(np.diff(center_y, axis=0), 0, 0)
    ds = np.sqrt(np.square(d_center_x) + np.square(d_center_y))
    arclength = np.sum(ds)
    return arclength


def spline_progress(path: np.ndarray, point: np.ndarray):
    # parametrize curve
    _, n = path.shape
    n_t = 500
    t = np.linspace(0, n, n_t, endpoint=True)
    xxyy = []
    for t_val in t:
        b = compute_bspline_bases(t_val, n)
        xx = np.dot(path[0, :], b)
        yy = np.dot(path[1, :], b)
        xxyy.append([xx, yy])
    xxyy = np.asarray(xxyy)
    # find minimum
    eucl_dist = np.linalg.norm(point - xxyy, ord=2, axis=1)
    min_dist = np.amin(eucl_dist)
    min_idx = np.where(eucl_dist == min_dist)[0]
    progress = t[min_idx] / n
    return progress


def compute_bspline_bases(x, n):
    x = max(x, 0)
    x = min(x, n - 2)
    # position in basis function
    v = np.zeros((n, 1))
    b = np.zeros((n, 1))
    for i in range(n):
        v[i, 0] = x - i + 2
        vv = v[i, 0]
        if vv < 0:
            b[i, 0] = 0
        elif vv < 1:
            b[i, 0] = 0.5 * vv ** 2
        elif vv < 2:
            b[i, 0] = 0.5 * (-3 + 6 * vv - 2 * vv ** 2)
        elif vv < 3:
            b[i, 0] = 0.5 * (3 - vv) ** 2
        else:
            b[i, 0] = 0
    return b


if __name__ == "__main__":
    state = VehicleState(x=0, y=0, theta=np.pi / 2, vx=1, delta=0.2)
    traj = state2traj(state, 15, 20)
    state2 = VehicleState(x=-3, y=0, theta=np.pi/4, vx=1, delta=-0.2)
    traj2 = state2traj(state2, 15, 20)
    test = np.array([-1, 3]).reshape([2, 1])
    p = spline_progress(traj, test)
    print(p)
    s = spline_s(traj, test)
    print(s)
    tck, u = interpolate.splprep(traj, k=2)
    new_points = interpolate.splev(u, tck)

    path1 = linearize_spline(traj)
    path2 = linearize_spline(traj2)
    res, intersect = find_intersect(path1, path2)
    print(res, intersect)
    fig, ax = plt.subplots()
    ax.plot(path1[0, :], path1[1, :], 'ro', markersize=3)
    ax.plot(path2[0, :], path2[1, :], 'go', markersize=3)
    ax.plot(new_points[0], new_points[1], 'r-')
    ax.plot(test[0, :], test[1, :], 'bo')
    if res:
        ax.plot(intersect[0], intersect[1], 'y*')
    plt.show()
