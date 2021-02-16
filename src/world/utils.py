from math import isclose

import networkx as nx
from decimal import Decimal as D
import itertools as it
from typing import List, cast
import numpy as np


import duckietown_world as dw
from duckietown_world.geo.transforms import SE2Transform
from duckietown_world.world_duckietown.lane_segment import LaneSegment
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap

import geometry as geo
from driving_games.structures import SE2_disc
from world.skeleton_graph import get_skeleton_graph


"""
Collection of functions that handle the module DuckietownWorld
"""

LaneName = str
NodeName = str
SE2value = np.array
Lane = LaneSegment


def interpolate(lane: dw.LaneSegment, beta: float) -> dw.SE2Transform:
    """
    Interpolate along the centerline of a lane. Start: beta=0, End beta=1
    """
    lane_length = lane.get_lane_length() # get the length of the lane
    along_lane = beta * lane_length # get the corresponding position along the lane
    transform = interpolate_along_lane(lane=lane, along_lane=along_lane)
    return transform


def interpolate_n_points(lane: dw.LaneSegment, betas: List[float]) -> List[dw.SE2Transform]:
    """
    Get pose sequence as a SE2Transform along the center line of a lane, beta=0 start beta=1 end
    """
    msg = f"betas = {betas} have to be in ascending order to follow a lane"
    assert all(map(isclose, sorted(betas), betas)), msg  # check if values are ascending
    transforms = [interpolate(lane, beta) for beta in betas]
    return transforms


def interpolate_along_lane(lane: LaneSegment, along_lane: float) -> SE2Transform:
    """ Input: lane and 1D position along the lane. Output: Pose on the duckietown map """

    dw_beta = lane.beta_from_along_lane(along_lane=along_lane)  # get the beta in in the dw representation
    p = lane.center_point(dw_beta)  # get pose
    transform = dw.SE2Transform.from_SE2(p)
    return transform


def interpolate_along_lane_n_points(
    lane: LaneSegment,
    positions_along_lane: List[float]
) -> List[SE2Transform]:
    """
    Input: lane and sequence of 1D positions along the lane. Output: Pose sequence on the duckietown map
    """
    msg = f"Positions={positions_along_lane} have to be in ascending order to follow a lane"
    assert all(map(isclose, sorted(positions_along_lane), positions_along_lane)), msg
    transforms = [interpolate_along_lane(lane, along_lane) for along_lane in positions_along_lane]
    return transforms


def from_SE2_disc_to_SE2Transform(q: SE2_disc) -> SE2Transform:
    """
    Converts from SE2_disc representation to the SE2 Transform wrapper from duckietown world
    """
    x, y, theta_deg = map(float, q)  # does not work with decimals
    theta_rad = np.deg2rad(theta_deg)
    q_SE2 = geo.SE2_from_translation_angle(t=[x, y], theta=theta_rad)
    q_transformed = dw.SE2Transform.from_SE2(q_SE2)
    return q_transformed


def from_SE2Transform_to_SE2_disc(q: SE2Transform) -> SE2_disc:
    """
    Converts from the SE2 Transform wrapper from duckietown world to the SE2_disc representation
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
    """
    *t, theta_deg = map(float, q)
    theta_rad = np.deg2rad(theta_deg)
    return geo.SE2_from_translation_angle(t, theta_rad)


def from_SE2_to_SE2_disc(q: SE2value) -> SE2_disc:
    """
    Converts from the SE2 representation used in the module geometry to SE2_disc
    """
    t, theta_rad = geo.translation_angle_from_SE2(q)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    return (D(x), D(y), D(theta_deg))


def merge_lanes(lanes: List[LaneSegment]) -> Lane:
    """
    Merges a list of consecutive lane segments to one single unified lane segment
    """
    width = lanes[0].width
    # Make a list of all the control points, while making sure that the points that overlap are only taken once
    contr_points_lanes = list(
        it.chain(
            *[ls.control_points[:-1] if ls is not lanes[-1]
              else ls.control_points for ls in lanes]
        )
    )

    # Creating a unified lane segment
    merged_lane_segments = dw.LaneSegment(
        width=width, control_points=contr_points_lanes
    )
    return merged_lane_segments


def get_lane_segments(duckie_map: DuckietownMap, lane_names: List[LaneName]) -> List[LaneSegment]:
    """
    Given a list of names of consecutive lane segments in a duckietown map,
    return the corresponding lane segments
    """
    sk = get_skeleton_graph(duckie_map)  # get the skeleton graph
    map_lane_segments = sk.root2  # get the map with all the lane segments
    lane_segments = [cast(LaneSegment, map_lane_segments.children[lane_name]) for lane_name in lane_names]
    return lane_segments


def get_lane_from_node_sequence(m: DuckietownMap, node_sequence: List[NodeName]) -> Lane:
    """
    For a sequence of nodes e.g ['P13', 'P2',...'P12'] this function returns the shortest lane which follows the given
    node sequence.
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
    path = list(it.chain(
        *[_path[:-1] if _path is not path_sequence[-1]
          else _path for _path in path_sequence]
    ))

    # get the sequence of lanes names
    lane_names = _get_lanes(path=path, graph=topology_graph)

    # extract the lane segments
    lane_segments = [cast(LaneSegment, map_lane_segments.children[lane_name]) for lane_name in lane_names]

    # merge the lane segments to one lane
    lane = merge_lanes(lane_segments)

    return lane


def _get_lanes(path, graph):
    edges = zip(path[:-1], path[1:])
    lanes = []
    for a, b in edges:
        lane = graph.get_edge_data(a, b)[0]['lane']
        lanes.append(lane)
    return lanes


def get_pose_in_ref_frame(abs_pose: SE2_disc, ref: SE2_disc) -> SE2_disc:
    """
    Returns the pose of an object in the reference frame given
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
    Get the pose along a center lane
    """
    pose_SE2_transform = interpolate_along_lane(lane=lane, along_lane=float(along_lane))
    return from_SE2Transform_to_SE2_disc(pose_SE2_transform)


def get_SE2disc_in_ref_from_along_lane(ref: SE2_disc, lane: Lane, along_lane: D) -> SE2_disc:
    """
    Get the pose along a center lane in the coordinates of a reference frame
    """
    # Get the SE2 representation of the absolute pose
    abs_pose = get_SE2disc_from_along_lane(lane=lane, along_lane=along_lane)
    ref_pose = get_pose_in_ref_frame(abs_pose=abs_pose, ref=ref)
    return ref_pose


class LaneSegmentHashable(LaneSegment):
    """
        Wrapper class for a LaneSegment to make it hashable (make it usable for a frozen dataclass, e.g. a state)
    """
    @classmethod
    def initializor(cls, lane_segment: LaneSegment) -> "LaneSegmentHashable":
        ls_dict = lane_segment.__dict__
        return cls(**ls_dict)

    def __hash__(self):
        ctr_as_SE2_disc = it.chain(
            *[from_SE2Transform_to_SE2_disc(ctr) for ctr in self.control_points]
        )
        to_hash = *ctr_as_SE2_disc, self.width
        return hash(to_hash)

    # def __eq__(self, other):
    #     return hash(self) == hash(other)
