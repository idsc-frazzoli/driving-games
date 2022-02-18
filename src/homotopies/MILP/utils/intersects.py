import numpy as np
from typing import Dict, List, Tuple, Optional
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.geo import SE2_interpolate, relative_pose
from geometry import SE2value, translation_angle_from_SE2
from itertools import combinations
from homotopies import logger

vehicle_geometry = VehicleGeometry.default_car()


def find_intersects(trajs: Dict[PlayerName, DgSampledSequence[SE2value]]) -> Dict[PlayerName, Dict[PlayerName, float]]:
    """
    input:
    maps of agents to predicted trajectories M(Agent, SampledSequence(pose))
    output:
    maps of agents to (agent, intersection)pairs M(A, M(A', s)),
        s represents the curvilinear coordinate of the intersection of A and A' along the trajectory of A
    """
    intersects = {}
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        if player1 not in intersects.keys():  # initialize dict
            intersects[player1] = {}
        if player2 not in intersects.keys():
            intersects[player2] = {}

        path1 = traj2path(trajs[player1])  # extract (x,y) from sampled sequence
        path2 = traj2path(trajs[player2])
        intersects_w = find_intersect(path1, path2)  # find intersection point in world frame
        if len(intersects_w) != 0:  # todo: currently only record the first intersection if 2 paths intersect more than once
            intersect_s1 = compute_s(path1, intersects_w[0])  # intersection point in curvilinear frame
            intersect_s2 = compute_s(path2, intersects_w[0])
            intersects[player1][player2] = intersect_s1
            intersects[player2][player1] = intersect_s2
    return intersects


def traj2path(traj: DgSampledSequence[SE2value]) -> List[Tuple[float, float]]:
    """this function extracts states x,y from sampled sequence"""
    poses = traj.values
    path = []
    for i in range(len(poses)):
        t, _ = translation_angle_from_SE2(poses[i])
        path += [[t[0], t[1]]]
    return path


def find_intersect(path1: List[Tuple[float, float]], path2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    the function returns all intersection points of 2 paths,
    2 adjacent points of a path are assumed to be connected with straight line
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


def check_intersect(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> Optional[
    Tuple[float, float]]:
    """
    this function checks if the two line segments intersect
    line equation used here: Ax+By=C, where
        A = y2-y1
        B = x1-x2
        C=A*x1+B*y1
    input:
    line1: 2*2 list, [[x1,y1], [x2, y2]]
    line2: 2*2 list, [[x3,y3], [x4, y4]]
    output:
    intersect: None if no intersection, (x,y) if intersect at (x,y).
    """
    line1 = np.array(line1)
    line2 = np.array(line2)
    if max(line1[:, 0]) < min(line2[:, 0]) or max(line2[:, 0]) < min(line1[:, 0]) or \
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
    buffer = 0.0001
    if max(line1[:, 0]) + buffer >= x >= min(line1[:, 0]) - buffer and max(line1[:, 1]) + buffer >= y >= min(
            line1[:, 1]) - buffer \
            and max(line2[:, 0]) + buffer >= x >= min(line2[:, 0]) - buffer and max(line2[:, 1]) + buffer >= y >= min(
        line2[:, 1]) - buffer:
        return x, y
    else:
        return None  # intersection point not on the segments


def compute_s(path: List[Tuple[float, float]], point: Tuple[float, float]) -> float:
    """
    this function computes the s coord of a given point on the path
    input:
    -path: N*2 list
    -point: coords in world frame
    output:
    -s: s coord of the point along the path
    """
    p1_idx, ratio = find_p_idx(path, point)
    s = 0
    path = np.array(path)
    for i in range(p1_idx):
        s += np.linalg.norm(path[i] - path[i + 1])
    s += ratio * np.linalg.norm(path[p1_idx + 1] - path[p1_idx])
    return s


def find_p_idx(path: List[Tuple[float, float]], point: Tuple[float, float]) -> Tuple[int, float]:
    """
    this function locates the point on the path
    input:
    -path: N*2 list
    -point: coords in world frame
    output:
    -p1_idx: the smaller index of the line segment where point is on
    -ratio: the fraction of the point from p1
    """
    length = len(path)
    path = np.array(path)
    point = np.array(point)
    eucl_dist = np.linalg.norm(point - path, ord=2, axis=1)
    min_dist = np.amin(eucl_dist)
    min_idx = np.where(eucl_dist == min_dist)[0][0]
    if min_idx == length - 1:
        p1_idx = min_idx - 1
        p2_idx = min_idx
    elif min_idx == 0:
        p1_idx = min_idx
        p2_idx = min_idx + 1
    elif max(path[min_idx:min_idx + 2, 0]) >= point[0] >= min(path[min_idx:min_idx + 2, 0]) and \
            max(path[min_idx:min_idx + 2, 1]) >= point[1] >= min(path[min_idx:min_idx + 2, 0]):
        # check whether point is on line(p_{idx-1}, p_idx) or (p_idx, p_{idx+1})
        p1_idx = min_idx
        p2_idx = min_idx + 1
    else:
        p1_idx = min_idx - 1
        p2_idx = min_idx
    ratio = np.linalg.norm(point - path[p1_idx]) / np.linalg.norm(path[p2_idx] - path[p1_idx])
    return p1_idx, ratio


def pose_from_s(traj: DgSampledSequence[SE2value], s: float) -> SE2value:
    """
    extract the pose along a trajectory at longitudinal coord s by interpolation
    """
    path = traj2path(traj)
    path = np.array(path)
    temp_s = 0
    idx = 0
    while temp_s < s:
        if idx >= path.shape[0] - 1:
            logger.info(f"Vehicle exceeds reference path!")
            return traj.at(traj.get_end())
        temp_s += np.linalg.norm(path[idx] - path[idx + 1])
        idx += 1
    ratio = (temp_s - s) / np.linalg.norm(path[idx - 1] - path[idx])
    pose1 = traj.values[idx]
    pose2 = traj.values[idx - 1]
    pose = SE2_interpolate(pose1, pose2, ratio)
    return pose


def get_box_size(pose1: SE2value, pose2: SE2value) -> Tuple[float, float]:
    """get widths of the constraint box along s1 and s2 axis"""
    pose_21 = relative_pose(pose1, pose2)
    _, theta_21 = translation_angle_from_SE2(pose_21)
    sinth = abs(np.sin(theta_21))
    tanth = abs(np.tan(theta_21))
    w1 = vehicle_geometry.width
    w2 = vehicle_geometry.width
    w_s1 = w1 / tanth + w2 / sinth + vehicle_geometry.length
    w_s2 = w2 / tanth + w1 / sinth + vehicle_geometry.length
    return w_s1, w_s2


def get_box(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
            intersects: Dict[PlayerName, Dict[PlayerName, SE2value]],
            player1: PlayerName,
            player2: PlayerName,
            buffer: float = 1) -> Tuple[Tuple[float, float], float, float]:
    """get the center coordinates and widths of the constraint box in s1-s2 frame"""
    s12 = intersects[player1][player2]
    s21 = intersects[player2][player1]
    pose12 = pose_from_s(trajs[player1], s12)
    pose21 = pose_from_s(trajs[player2], s21)
    w_s12, w_s21 = get_box_size(pose12, pose21)
    return (s12, s21), w_s12 * buffer, w_s21 * buffer


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """compute the length of a given path"""
    s = 0
    length = len(path)
    path = np.array(path)
    for i in range(length - 1):
        s += np.linalg.norm(path[i] - path[i + 1])
    return s


def get_s_max(trajs: Dict[PlayerName, DgSampledSequence[SE2value]]) -> Dict[PlayerName, float]:
    """compute the length of each path"""
    s_max = {}
    for player in trajs.keys():
        path = traj2path(trajs[player])
        s_max[player] = compute_path_length(path)
    return s_max
