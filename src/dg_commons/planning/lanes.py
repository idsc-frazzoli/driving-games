from dataclasses import dataclass
from math import isclose, pi, atan
from typing import Sequence, List

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from duckietown_world import SE2Transform, relative_pose
from duckietown_world.utils import memoized_reset, SE2_interpolate, SE2_apply_R2
from duckietown_world.world_duckietown.lane_segment import get_distance_two
from geometry import SO2value, SO2_from_angle, SE2_from_translation_angle, SE2value, SE2, \
    translation_angle_scale_from_E2, translation_angle_from_SE2, T2value
from scipy.optimize import minimize_scalar
from zuper_commons.types import ZValueError


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

    def lane_pose_from_SE2_generic(self, q: SE2value, tol: float = 1e-7) -> DgLanePose:
        p, _, _ = translation_angle_scale_from_E2(q)

        beta, q0 = self.find_along_lane_closest_point(p, tol=tol)
        along_lane = self.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)

        r, relative_heading, _ = translation_angle_scale_from_E2(rel)
        lateral = r[1]

        return self.lane_pose(along_lane=along_lane, relative_heading=relative_heading, lateral=lateral)

    def find_along_lane_closest_point(self, p: T2value, tol: float = 1e-7):
        def get_delta(beta):
            q0 = self.center_point(beta)
            t0, _ = translation_angle_from_SE2(q0)
            d = np.linalg.norm(p - t0)

            d1 = np.array([0, -d])
            p1 = SE2_apply_R2(q0, d1)

            d2 = np.array([0, +d])
            p2 = SE2_apply_R2(q0, d2)

            D2 = np.linalg.norm(p2 - p)
            D1 = np.linalg.norm(p1 - p)
            res = np.maximum(D1, D2)
            return res

        bracket = (-1.0, len(self.control_points))
        res0 = minimize_scalar(get_delta, bracket=bracket, tol=tol)
        beta0 = res0.x
        q = self.center_point(beta0)
        return beta0, q

    def lane_pose(self, along_lane: float, relative_heading: float, lateral: float) -> DgLanePose:
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

    def radius(self, beta: float) -> float:
        n = len(self.control_points)
        i = int(np.floor(beta))
        if i < 0 or i >= n - 1:
            raise ZValueError("Lane width is not defined outside the Lane")
        else:
            alpha = beta - i
            r0 = self.control_points[i].r
            r1 = self.control_points[i + 1].r
            return r0 * (1 - alpha) + r1 * alpha

    def center_point(self, beta: float) -> SE2value:
        n = len(self.control_points)
        i = int(np.floor(beta))

        if i < 0:
            q0 = self.control_points[0].q.asmatrix2d().m
            q1 = SE2.multiply(q0, SE2_from_translation_angle([0.1, 0], 0))
            alpha = beta

        elif i >= n - 1:
            q0 = self.control_points[-1].q.asmatrix2d().m
            q1 = SE2.multiply(q0, SE2_from_translation_angle([0.1, 0], 0))
            alpha = beta - (n - 1)
        else:
            alpha = beta - i
            q0 = self.control_points[i].q.asmatrix2d().m
            q1 = self.control_points[i + 1].q.asmatrix2d().m
        q = SE2_interpolate(q0, q1, alpha)
        return q
