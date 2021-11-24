from functools import lru_cache
from math import isclose
from time import perf_counter

import networkx as nx
from decimal import Decimal as D
import itertools as it
from typing import List, cast, Tuple
import numpy as np

import duckietown_world as dw
from duckietown_world.geo.transforms import SE2Transform
from duckietown_world.utils import SE2_apply_R2
from duckietown_world.world_duckietown.lane_segment import LaneSegment, LanePose
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap

import geometry as geo
from scipy.optimize import minimize_scalar

from dg_commons import SE2_apply_T2
from driving_games.structures import SE2_disc
from _tmp._deprecated.world.skeleton_graph import get_skeleton_graph

"""
Collection of functions that handle the module DuckietownWorld
"""

LaneName = str
NodeName = str
SE2value = np.array
Lane = LaneSegment


def interpolate(lane: dw.LaneSegment, beta: float) -> dw.SE2Transform:
    """
    Normalized interpolation along the centerline of a lane. Start: beta=0, End beta=1

    :param lane: A lane segment (duckietown world class)
    :param beta: Parameter of interpolation, beta=0 means start of the lane, beta=1 means end of the lane
    :return: The SE2Transform representing the pose along the centerline
    """
    lane_length = lane.get_lane_length()  # get the length of the lane
    along_lane = beta * lane_length  # get the corresponding position along the lane
    transform = interpolate_along_lane(lane=lane, along_lane=along_lane)
    return transform


def interpolate_n_points(lane: dw.LaneSegment, betas: List[float]) -> List[dw.SE2Transform]:
    """
    Get pose sequence as a SE2Transform along the center line of a lane.
    Normalized interpolation given a sequence of betas. Start: beta=0, End: beta=1

    :param lane: A lane segment (duckietown world class)
    :param betas: An ascending list of normalized interpolation parameters. Start of lane: beta=0, End of lane: beta=1
    :return: A sequence of SE2Transform representing the pose along the center lane
    """
    msg = f"betas = {betas} have to be in ascending order to follow a lane"
    assert all(map(isclose, sorted(betas), betas)), msg  # check if values are ascending
    transforms = [interpolate(lane, beta) for beta in betas]
    return transforms


def interpolate_along_lane(lane: LaneSegment, along_lane: float) -> SE2Transform:
    """
    Interpolate along the centerline of a lane. If the lane has length 10,along_lane=0 gives the pose at the start
    of the lane and along_lane=10 gives the pose at the end of the lane.

    :param lane: A lane segment (duckietown world class)
    :param along_lane: Parameter of interpolation, which corresponds to the position along the lane.

    :return: The SE2Transform representing the pose along the centerline.
    """

    dw_beta = lane.beta_from_along_lane(along_lane=along_lane)  # get the beta in in the dw representation
    p = lane.center_point(dw_beta)  # get pose
    transform = dw.SE2Transform.from_SE2(p)
    return transform


def interpolate_along_lane_n_points(lane: LaneSegment, positions_along_lane: List[float]) -> List[SE2Transform]:
    """
    Get pose sequence as a SE2Transform along the center line of a lane. As an input you have to give a sequence in
    ascending order of the parameter of interpolation. They correspond to the position along the lane. If the lane has
    length 10,along_lane=0 gives the pose at the start of the lane and along_lane=10 gives the pose at the end of the lane.

    :param lane: A lane segment (duckietown world class)
    :param positions_along_lane: Sequence in ascending order of the parameter of interpolation.

    :return: The SE2Transform sequence representing the poses along the centerline.
    """
    msg = f"Positions={positions_along_lane} have to be in ascending order to follow a lane"
    assert all(map(isclose, sorted(positions_along_lane), positions_along_lane)), msg
    transforms = [interpolate_along_lane(lane, along_lane) for along_lane in positions_along_lane]
    return transforms


def from_SE2_disc_to_SE2Transform(q: SE2_disc) -> SE2Transform:
    """
    Converts from SE2_disc representation to the SE2 Transform wrapper from duckietown world

    :param q: The pose in SE2_disc representation from the driving-game module
    :return: The pose in the SE2Transform representation duckietown-world module
    """
    x, y, theta_deg = map(float, q)  # does not work with decimals
    theta_rad = np.deg2rad(theta_deg)
    q_SE2 = geo.SE2_from_translation_angle(t=[x, y], theta=theta_rad)
    q_transformed = dw.SE2Transform.from_SE2(q_SE2)
    return q_transformed


def from_SE2Transform_to_SE2_disc(q: SE2Transform) -> SE2_disc:
    """
    Converts from the SE2 Transform wrapper from duckietown world to the SE2_disc representation

    :param q: The pose in the SE2Transform representation duckietown-world module
    :return: The pose in SE2_disc representation from the driving-game module
    """
    q_SE2 = dw.SE2Transform.as_SE2(q)
    t, theta_rad = geo.translation_angle_from_SE2(q_SE2)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    se2_disc = (D(x), D(y), D(theta_deg))
    return se2_disc


def from_SE2_disc_to_SE2(q: SE2_disc) -> SE2value:
    """
    Converts from SE2_disc to the SE2 representation used in the module geometry

    :param q: The pose in SE2_disc representation from the driving-game module
    :return: The pose in SE2value representation from the geometry module
    """
    *t, theta_deg = map(float, q)
    theta_rad = np.deg2rad(theta_deg)
    return geo.SE2_from_translation_angle(t, theta_rad)


def from_SE2_to_SE2_disc(q: SE2value) -> SE2_disc:
    """
    Converts from the SE2 representation used in the module geometry to a SE2_disc

    :param q: The pose in SE2value representation from the geometry module
    :return: The pose in SE2_disc representation from the driving-game module
    """
    t, theta_rad = geo.translation_angle_from_SE2(q)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    return (D(x), D(y), D(theta_deg))


def merge_lanes(lanes: List[LaneSegment]) -> Lane:
    """
    Merges a list of consecutive lane segments to one single unified lane segment

    :param lanes: List of consecutive lane segments from a duckietown map.
    :return: One single lane segment
    """
    width = lanes[0].w_half
    # Make a list of all the control points, while making sure that the points that overlap are only taken once
    contr_points_lanes = list(
        it.chain(*[ls.control_points[:-1] if ls is not lanes[-1] else ls.control_points for ls in lanes])
    )

    # Creating a unified lane segment
    merged_lane_segments = dw.LaneSegment(width=width, control_points=contr_points_lanes)
    return merged_lane_segments


def get_lane_segments(duckie_map: DuckietownMap, lane_names: List[LaneName]) -> List[LaneSegment]:
    """
    Given a list of names of consecutive lane segments in a duckietown map (seen in the map network
    found in the maps folder) it returns a list of the corresponding lane segments.

    :param duckie_map: A duckietown map containing the lanes which should be extracted
    :param lane_names: A list of the names of the lane segments as indicated in the map network found in the maps folder
    :return: The list of the lane segments in the same order as the list of names.
    """
    sk = get_skeleton_graph(duckie_map)  # get the skeleton graph
    map_lane_segments = sk.root2  # get the map with all the lane segments
    lane_segments = [cast(LaneSegment, map_lane_segments.children[lane_name]) for lane_name in lane_names]
    return lane_segments


def get_lane_from_node_sequence(m: DuckietownMap, node_sequence: List[NodeName]) -> Lane:
    """
    For a sequence of nodes e.g ['P13', 'P2',...'P12'] this function returns the shortest lane which follows the given
    node sequence.

    :param m: A duckietown map
    :param node_sequence: A list of names of the nodes to follow (indicated in the map network found in the maps folder)
    :return: One lane that follows the node sequence

    """
    assert len(node_sequence) > 1, "At least two nodes must be given"

    sk = get_skeleton_graph(m)  # get the skeleton graph
    topology_graph = sk.G
    map_lane_segments = sk.root2  # get the map with all the lane segments

    # Extract the partial paths from one node to another
    path_sequence = [
        nx.shortest_path(topology_graph, start, end) for start, end in zip(node_sequence[:-1], node_sequence[1:])
    ]

    # remove the nodes at the end of the partial paths such that their are only present once
    path = list(it.chain(*[_path[:-1] if _path is not path_sequence[-1] else _path for _path in path_sequence]))

    # get the sequence of lanes names
    lane_names = _get_lanes(path=path, graph=topology_graph)

    # extract the lane segments
    lane_segments = [cast(LaneSegment, map_lane_segments.children[lane_name]) for lane_name in lane_names]

    # merge the lane segments to one lane
    lane = merge_lanes(lane_segments)

    return lane


def _get_lanes(path, graph):
    """
    Get the names of the lane following a certain path (nodes in the network found in the maps folder)
    """
    edges = zip(path[:-1], path[1:])
    lanes = []
    for a, b in edges:
        lane = graph.get_edge_data(a, b)[0]["lane"]
        lanes.append(lane)
    return lanes


def get_pose_in_ref_frame(abs_pose: SE2_disc, ref: SE2_disc) -> SE2_disc:
    """
    Returns the pose of an object in the reference frame given

    :param abs_pose: The pose in the absolute coordinate system as an SE2_disc
    :param ref: The pose of the reference frame in the absolute coordinate system
    :return: The pose in the reference coordinate system given
    """
    *t_abs, theta_abs_deg = map(float, abs_pose)
    theta_abs_rad = np.deg2rad(theta_abs_deg)
    q_abs = geo.SE2_from_translation_angle(t_abs, theta_abs_rad)

    # Get SE2 representation of the ref pose
    *t_ref, theta_ref_deg = map(float, ref)
    theta_ref_rad = np.deg2rad(theta_ref_deg)
    q_ref = geo.SE2_from_translation_angle(t_ref, theta_ref_rad)

    # Get the the pose of the duckie in the reference frame
    q_abs_from_q_ref = geo.SE2.multiply(geo.SE2.inverse(q_ref), q_abs)
    t, theta_rad = geo.translation_angle_from_SE2(q_abs_from_q_ref)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    return (D(x), D(y), D(theta_deg))


def get_SE2disc_from_along_lane(lane: Lane, along_lane: D) -> SE2_disc:
    """
    Get the pose as a SE2_disc along a center lane

    :param lane: The lane one wants to follow
    :param along_lane: The position along the lane
    :return: The pose as a SE2_disc along the lane in the coordinate system of the lane.
    """
    pose_SE2_transform = interpolate_along_lane(lane=lane, along_lane=float(along_lane))
    return from_SE2Transform_to_SE2_disc(pose_SE2_transform)


def get_SE2disc_in_ref_from_along_lane(ref: SE2_disc, lane: Lane, along_lane: D) -> SE2_disc:
    """
    Get the pose along a center lane in the coordinates of a reference frame

    :param lane: The lane one wants to follow
    :param ref: The pose of the reference frame in the coordinate system of the lane.
    :param along_lane: The position along the lane
    :return: The pose as a SE2_disc along the lane in the given reference coordinate system.
    """
    # Get the SE2 representation of the absolute pose
    abs_pose = get_SE2disc_from_along_lane(lane=lane, along_lane=along_lane)
    ref_pose = get_pose_in_ref_frame(abs_pose=abs_pose, ref=ref)
    return ref_pose


class LaneSegmentHashable(LaneSegment):
    """
    Wrapper class for a LaneSegment to make it hashable (make it usable for a frozen dataclass, e.g. a state)
    """

    time: float = 0.0

    @classmethod
    def initializor(cls, lane_segment: LaneSegment) -> "LaneSegmentHashable":
        """
        Creates a hashable lane segment given a LaneSegment from the duckietown-world module

        :param lane_segment: A duckietown world lane segment
        :returns: A hashable version of the lane segment given
        """
        ls_dict = lane_segment.__dict__
        return cls(**ls_dict)

    def __hash__(self):
        """
        Hash function for the lane segment. The control points and the width of the lane get hashed.
        """
        ctr_as_SE2_disc = it.chain(*[from_SE2Transform_to_SE2_disc(ctr) for ctr in self.control_points])
        to_hash = *ctr_as_SE2_disc, self.width
        return hash(to_hash)

    @lru_cache(None)
    def lane_pose_from_SE2Transform(self, qt: SE2Transform, tol=0.001) -> LanePose:
        tic = perf_counter()
        lane_pose = LaneSegment.lane_pose_from_SE2Transform(self, qt=qt, tol=tol)
        LaneSegmentHashable.time += perf_counter() - tic
        return lane_pose

    def find_along_lane_closest_point(self, p, tol=0.001) -> Tuple[float, geo.SE2value]:
        def get_delta(beta):
            q0 = self.center_point(beta)
            t0, _ = geo.translation_angle_from_SE2(q0)
            d = np.linalg.norm(p - t0)

            d1 = np.array([0, -d])
            p1 = SE2_apply_T2(q0, d1)

            d2 = np.array([0, +d])
            p2 = SE2_apply_T2(q0, d2)

            D2 = np.linalg.norm(p2 - p)
            D1 = np.linalg.norm(p1 - p)
            res = np.maximum(D1, D2)
            return res

        bracket = (-1.0, len(self.control_points))
        res0 = minimize_scalar(get_delta, bracket=bracket, tol=tol)
        beta0 = res0.x
        q = self.center_point(beta0)
        return beta0, q
