import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.utils import extract_pose_from_state
from dg_commons.geo import SE2_interpolate
from geometry import SE2value, translation_angle_from_SE2
from itertools import combinations

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()


def predict(obs: Dict[PlayerName, VehicleState]) -> Dict[PlayerName, DgSampledSequence[SE2value]]:
    """
    input:
    -obs: maps of agents to their current states
    output:
    maps of agents to their predicted trajectories(DgSampledSequence of SE2 poses)
    """
    trajs = {}
    for player in obs.keys():
        state = obs[player]
        traj = state2traj(state, 20, 0.2)  # arguements: current state, predicted time(s), time step
        trajs[player] = traj
    return trajs


def state2traj(state: VehicleState, horizon: float, dt: float) -> DgSampledSequence[SE2value]:
    """
    This function takes the current state of an object as input and outputs the predicted trajectory
    assuming constant vx and delta(no control inputs)
    input:
    -state: VehicleState(x,y,theta,vx,delta)
    -horizon: duration of the predicted trajectory
    -dt: time step used for prediction
    output:
    -traj: DGSampledSequence of SE2value
    """
    n = int(horizon / dt)
    curr_pose = extract_pose_from_state(state)
    traj = [curr_pose]
    time = [0]
    v = state.vx
    delta = state.delta
    for i in range(n - 1):
        t, theta = translation_angle_from_SE2(traj[i])
        dtheta = v * np.tan(delta) / vehicle_geometry.length
        vy = dtheta * vehicle_geometry.lr
        curr_state = VehicleState(x=t[0] + dt * (v * np.cos(theta) - vy * np.sin(theta)),
                                  y=t[1] + dt * (v * np.sin(theta) + vy * np.cos(theta)),
                                  theta=theta + dt * dtheta,
                                  vx=v,
                                  delta=delta)
        curr_pose = extract_pose_from_state(curr_state)
        time += [time[i] + dt]
        traj += [curr_pose]
    return DgSampledSequence[SE2value](time, values=traj)


def find_intersects(trajs: Dict[PlayerName, DgSampledSequence[SE2value]]) -> Dict[PlayerName, Dict[PlayerName, float]]:
    """
    input:
    maps of agents to predicted trajectories M(A, SampledSequence(pose))
    output:
    maps of agents to (agent, intersection)pairs M(A, M(A', s)),
        s represents the curvilinear coordinate of the intersection of A and A' along A's trajectory
    """
    intersects = {}
    for player_pair in combinations(trajs.keys(),2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        if not player1 in intersects.keys():
            intersects[player1] = {}
        if not player2 in intersects.keys():
            intersects[player2] = {}

        path1 = traj2path(trajs[player1])
        path2 = traj2path(trajs[player2])
        intersects_w = find_intersect(path1, path2)  # intersect point in world frame
        print(intersects_w)
        if len(intersects_w) != 0:
            for intersect in intersects_w:  # todo: currently only record the last intersection if 2 paths intersect more than once
                intersect_s1 = compute_s(path1, intersect)  # intersect point in s coord
                intersect_s2 = compute_s(path2, intersect)
                intersects[player1][player2] = intersect_s1
                intersects[player2][player1] = intersect_s2
    return intersects


def traj2path(traj: DgSampledSequence[SE2value]) -> List[Tuple[float, float]]:
    """this function extracts states x,y and theta from sampled sequence"""
    poses = traj.values
    path = []
    for i in range(len(poses)):
        t, _ = translation_angle_from_SE2(poses[i])
        path += [[t[0], t[1]]]
    return path


def find_intersect(path1: List[Tuple[float, float]], path2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    the function returns all intersection points of 2 point sequences in world frame,
    2 adjacent points are assumed to be connected with straight line
    input:
    -path1: N1*2 list
    -path2: N2*2 list
    output:
    -intersect: n*2 list
    """
    len1 = len(path1)
    len2 = len(path2)
    intersect = []
    if len1 < 3 and len2 < 3:
        res = check_intersect(path1, path2)
        if res is None:
            return []
        else:
            return [res]
    else:
        if len1 < 3:
            path21 = path2[0:int(len2 / 2) + 1]
            path22 = path2[int(len2 / 2):]
            intersect += find_intersect(path1, path21)
            intersect += find_intersect(path1, path22)
        elif len2 < 3:
            path11 = path1[0:int(len1 / 2) + 1]
            path12 = path1[int(len1 / 2):]
            intersect += find_intersect(path11, path2)
            intersect += find_intersect(path12, path2)
        else:
            path11 = path1[0:int(len1 / 2) + 1]
            path12 = path1[int(len1 / 2):]
            path21 = path2[0:int(len2 / 2) + 1]
            path22 = path2[int(len2 / 2):]
            intersect += find_intersect(path11, path21)
            intersect += find_intersect(path11, path22)
            intersect += find_intersect(path12, path21)
            intersect += find_intersect(path12, path22)
        return intersect


def check_intersect(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    this function checks if the two input line segments intersect
    A = y2-y1
    B = x1-x2
    C=A1*x1+B1*y1
    line func: Ax+By=C
    input:
    line1: 2*2 list, [[x1,y1], [x2, y2]]
    line2: 2*2 list, [[x3,y3], [x4, y4]]
    output:
    intersect: None if no intersection, [[x,y]] if intersect at (x,y).
    """
    line1 = np.array(line1)
    line2 = np.array(line2)
    if max(line1[:, 0]) < min(line2[:, 0]) or max(line2[:, 0]) < min(line1[:, 0]) or\
            max(line1[:, 1]) < min(line2[:, 1]) or max(line2[:, 1]) < min(line1[:, 1]):
        return None  # 2segments don't have mutual interval
    A1 = line1[1, 1] - line1[0, 1]
    B1 = line1[0, 0] - line1[1, 0]
    C1 = A1 * line1[0, 0] + B1 * line1[0, 1]
    A2 = line2[1, 1] - line2[0, 1]
    B2 = line2[0, 0] - line2[1, 0]
    C2 = A2 * line2[0, 0] + B2 * line2[0, 1]
    det = A1 * B2 - A2 * B1
    if det == 0:  # two lines are parallel
        return None
    else:
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
    if max(line1[:, 0]) >= x >= min(line1[:, 0]) and max(line1[:, 1]) >= y >= min(line1[:, 1]) \
            and max(line2[:, 0]) >= x >= min(line2[:, 0]) and max(line2[:, 1]) >= y >= min(line2[:, 1]):
        return x, y
    else:
        return None  # intersection point not on the segments


def compute_s(path: List[Tuple[float, float]], point: Tuple[float, float]) -> float:
    p1_idx, ratio = find_p_idx(path, point)
    s = 0
    path = np.array(path)
    for i in range(p1_idx):
        s += np.linalg.norm(path[i]-path[i+1])
    s += ratio * np.linalg.norm(path[p1_idx+1]-path[p1_idx])
    return s


def find_p_idx(path: List[Tuple[float, float]], point: Tuple[float, float])-> Tuple[int, float]:
    length = len(path)
    path = np.array(path)
    point = np.array(point)
    eucl_dist = np.linalg.norm(point - path, ord=2, axis=1)
    min_dist = np.amin(eucl_dist)
    min_idx = np.where(eucl_dist == min_dist)[0][0]
    if min_idx==length-1:
        p1_idx = min_idx-1
        p2_idx = min_idx
    elif min_idx==0:
        p1_idx = min_idx
        p2_idx = min_idx+1
    elif max(path[min_idx:min_idx + 2, 0]) >= point[0] >= min(path[min_idx:min_idx + 2, 0]) and \
            max(path[min_idx:min_idx + 2, 1]) >= point[1] >= min(path[min_idx:min_idx + 2, 0]):
        # check whether point is on line(p_idx-1, p_idx) or (p_idx, p_idx+1)
        p1_idx = min_idx
        p2_idx = min_idx + 1
    else:
        p1_idx = min_idx - 1
        p2_idx = min_idx
    ratio = (point[0] - path[p1_idx, 0]) / (path[p2_idx, 0] - path[p1_idx, 0])
    return p1_idx, ratio


def pose_from_s(traj: DgSampledSequence[SE2value], s: float) -> SE2value:
    path=traj2path(traj)
    path=np.array(path)
    temp_s = 0
    idx=0
    while temp_s < s:
        temp_s += np.linalg.norm(path[idx]-path[idx+1])
        idx += 1
    ratio = (temp_s-s)/np.linalg.norm(path[idx-1]-path[idx])
    pose1 = traj.values[idx]
    pose2 = traj.values[idx-1]
    pose = SE2_interpolate(pose1, pose2, ratio)
    return pose


if __name__ == "__main__":
    state1 = VehicleState(x=0, y=3, theta=np.pi / 2, vx=1, delta=-0.4)
    # traj1 = state2traj(state1, 15, 0.2)
    state2 = VehicleState(x=-5, y=0, theta=np.pi / 4, vx=1, delta=-0.1)
    # traj2 = state2traj(state2, 15, 0.2)
    test = np.array([-1, 3]).reshape([2, 1])
    player1 = PlayerName('p1')
    player2 = PlayerName('p2')
    obs = {player1: state1, player2: state2}

    trajs = predict(obs)
    intersects = find_intersects(trajs)
    print(intersects)
    traj1 = trajs[player1]
    traj2 = trajs[player2]
    fig, ax = plt.subplots()
    path1=np.array(traj2path(traj1))
    path2 = np.array(traj2path(traj2))
    ax.plot(path1[:, 0], path1[:, 1], 'ro-', markersize=3)
    ax.plot(path2[:, 0], path2[:, 1], 'go-', markersize=3)
    # ax.plot(new_points[0], new_points[1], 'r-')
    # if len(intersects_w) > 0:
    #     intersect = np.array(intersects_w)
    #     ax.plot(intersect[:, 0], intersect[:, 1], 'y*')
    plt.show()
