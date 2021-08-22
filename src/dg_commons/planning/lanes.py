from dataclasses import dataclass
from math import isclose, pi, atan
from typing import Sequence, List

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from duckietown_world import SE2Transform
from duckietown_world.utils import memoized_reset
from duckietown_world.world_duckietown.lane_segment import get_distance_two
from geometry import SO2value, SO2_from_angle


@dataclass(unsafe_hash=True)
class DgLanePose:
    """ Very detailed information about the "position in the lane". """

    # am I "inside" the lane?
    inside: bool
    # if not, am I outside on the left or right?
    outside_left: bool
    outside_right: bool
    # am I "inside" considering the longitudinal position?
    along_inside: bool
    # if not, am I outside before? (along_lane < 0)
    along_before: bool  #
    # or am I outside after? (along_lane > L)
    along_after: bool

    # Lateral, where 0 = lane center, positive to the left
    lateral: float
    # Longitudinal, along the lane. Starts at 0, positive going forward
    along_lane: float
    # Heading direction: 0 means aligned with the direction of the lane
    relative_heading: float

    # am I going in the right direction?
    correct_direction: bool
    # lateral position of closest left lane boundary
    lateral_left: float
    # lateral position of closest right lane boundary
    lateral_right: float

    # The distance from us to the left lane boundary
    distance_from_left: float
    # The distance from us to the right lane boundary
    distance_from_right: float
    # The distance from us to the center = abs(lateral)
    distance_from_center: float
    # center_point: anchor point on the center of the lane
    center_point: SE2Transform


@dataclass
class LaneCtrPoint:
    q: SE2Transform
    """ The centerline control point in SE2"""
    r: float
    """ The radius (half width) at the corresponding centerline control point"""


_rot90: SO2value = SO2_from_angle(pi / 2)


class DgLanelet:
    # todo taking the best from commonroad Lanelet and Lanesegments
    def __init__(self, control_points: Sequence[LaneCtrPoint]):
        self.control_points: List[LaneCtrPoint] = list(control_points)

    @classmethod
    def from_commonroad_lanelet(cls, lanelet: Lanelet) -> "DgLanelet":
        left_vertices = lanelet.left_vertices
        right_vertices = lanelet.right_vertices
        ctr_points = []
        for i, center in enumerate(lanelet.center_vertices):
            normal = right_vertices[i] - left_vertices[i]
            tangent = _rot90 @ normal
            theta = atan(tangent)  # todo check maybe atan2?
            q = SE2Transform(p=center, theta=theta)
            ctr_points.append(LaneCtrPoint(
                q, r=np.linalg.norm(normal) / 2
            ))
        return DgLanelet(ctr_points)

    @memoized_reset
    def get_lane_lengths(self) -> List[float]:
        res = []
        for i in range(len(self.control_points) - 1):
            p0 = self.control_points[i].q
            p1 = self.control_points[i + 1].q
            sd = get_distance_two(p0.as_SE2(), p1.as_SE2())
            res.append(sd)
        return res

    def get_lane_length(self) -> float:
        return sum(self.get_lane_lengths())

    def lane_pose_from_SE2(self) -> DgLanePose:
        # todo
        pass

    def along_lane_from_beta(self, beta: float) -> float:
        """ Returns the position along the lane (parametrized in distance)"""
        lengths = self.get_lane_lengths()
        if beta < 0:
            return beta
        elif beta >= len(self.control_points) - 1:
            rest = beta - (len(self.control_points) - 1)
            return sum(lengths) + rest
        else:
            i = int(np.floor(beta))
            rest = beta - i
            res = sum(lengths[:i]) + lengths[i] * rest
            return res

    def beta_from_along_lane(self, along_lane: float) -> float:
        """Returns the progress along the lane (parametrized in control points)"""
        lengths = self.get_lane_lengths()
        x0 = along_lane
        n = len(self.control_points)
        S = sum(lengths)

        if x0 < 0:
            beta = x0
            return beta
        elif x0 > S:
            beta = (n - 1.0) + (x0 - S)
            return beta
        elif isclose(x0, S, abs_tol=1e-8):
            beta = n - 1.0
            return beta
        assert 0 <= x0 < S, (x0, S)

        for i in range(n - 1):
            start_x = sum(lengths[:i])
            end_x = sum(lengths[: i + 1])
            if start_x <= x0 < end_x:
                beta = i + (x0 - start_x) / lengths[i]
                return beta
        assert False
