from math import isclose
from decimal import Decimal as D
import itertools as it

from typing import List, cast
import numpy as np

import geometry as geo
from driving_games.structures import SE2_disc
import duckietown_world as dw
from duckietown_world.geo.transforms import SE2Transform
from duckietown_world.world_duckietown.lane_segment import LaneSegment
from duckietown_world.world_duckietown.duckietown_map import DuckietownMap

"""
Collection of functions that handle the module DuckietownWorld
"""

LaneName = str
SE2value = np.array


def interpolate(lane: LaneSegment, beta: float) -> SE2Transform:
    """ Interpolate along the centerline of a lane. Start: beta=0, End beta=1 """

    n_ctr_points = len(lane.control_points)  # get the control points of the lane
    dw_beta = beta * (n_ctr_points - 1)  # transform the beta to the beta used by duckietown world
    p = lane.center_point(dw_beta)  # get the pose
    transform = dw.SE2Transform.from_SE2(p)  # transform the pose
    return transform


def interpolate_n_points(lane: dw.LaneSegment, betas: List[float]) -> List[dw.SE2Transform]:
    """ Get pose sequence as a SE2Transform along the center line of a lane, beta=0 start beta=1 end """

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
    """ Input: lane and sequence of 1D positions along the lane. Output: Pose sequence on the duckietown map """

    msg = f"Positions={positions_along_lane} have to be in ascending order to follow a lane"
    assert all(map(isclose, sorted(positions_along_lane), positions_along_lane)), msg
    transforms = [interpolate_along_lane(lane, along_lane) for along_lane in positions_along_lane]
    return transforms


def from_SE2_disc_to_SE2Transform(q: SE2_disc) -> SE2Transform:
    """ Converts from SE2_disc representation to the SE2 Transform wrapper from duckietown world"""

    x, y, theta_deg = map(float, q)  # does not work with decimals
    theta_rad = np.deg2rad(theta_deg)
    q_SE2 = geo.SE2_from_translation_angle(t=[x, y], theta=theta_rad)
    q_transformed = dw.SE2Transform.from_SE2(q_SE2)
    return q_transformed


def from_SE2Transform_to_SE2_disc(q: SE2Transform) -> SE2_disc:
    """ Converts from the SE2 Transform wrapper from duckietown world to the SE2_disc representation """

    q_SE2 = dw.SE2Transform.as_SE2(q)
    t, theta_rad = geo.translation_angle_from_SE2(q_SE2)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    se2_disc = (D(x), D(y), D(theta_deg))
    return se2_disc


def from_SE2_disc_to_SE2(q: SE2_disc) -> SE2value:
    *t, theta_deg = map(float, q)
    theta_rad = np.deg2rad(theta_deg)
    return geo.SE2_from_translation_angle(t, theta_rad)


def from_SE2_to_SE2_disc(q: SE2value) -> SE2_disc:
    t, theta_rad = geo.translation_angle_from_SE2(q)
    x, y = t
    theta_deg = np.rad2deg(theta_rad)
    return (D(x), D(y), D(theta_deg))


def merge_lanes(lanes: List[LaneSegment]) -> LaneSegment:
    """ Merges a list of consecutive lane segments to one single unified lane segment """

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
    sk = dw.get_skeleton_graph(duckie_map)  # get the skeleton graph
    map_lane_segments = sk.root2  # get the map with all the lane segments
    lane_segments = [cast(LaneSegment, map_lane_segments.children[lane_name]) for lane_name in lane_names]
    return lane_segments


class DuckietownMapHashable(DuckietownMap):
    """
    Wrapper class for a DuckietownMap to make it hashable (make it usable for a DuckieState)
    """

    def __hash__(self):
        return hash(repr(self))

    # def __eq__(self, other):
    #     return hash(self) == hash(other)

    @classmethod
    def initializor(cls, duckie_map: DuckietownMap):
        dm_dict = duckie_map.__dict__
        return cls(**dm_dict)


class LaneSegmentHashable(LaneSegment):
    """
    Wrapper class for a LaneSegment to make it hashable (make it usable for a DuckieState)
    """

    def __hash__(self):
        return hash(repr(self))

    # def __eq__(self, other):
    #     return hash(self) == hash(other)

    @classmethod
    def initializor(cls, lane_segment: LaneSegment):
        ls_dict = lane_segment.__dict__
        return cls(**ls_dict)
