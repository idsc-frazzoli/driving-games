import math
from dataclasses import dataclass
from typing import Optional
from sim.models.vehicle_structures import VehicleGeometry

import numpy as np
import scipy.optimize
from geometry import SE2value, SE2_from_translation_angle, translation_angle_from_SE2
from dg_commons.planning.lanes import DgLanelet
from games import X, U
import scipy.linalg

__all__ = ["LQR", "LQRParam"]


def lqr(a, b, q, r):
    """Solve the continuous time lqr controller.

    dx/dt = ax + bu

    cost = integral x.T*q*x + u.T*r*u
    """
    # first, try to solve the ricatti equation
    x = np.matrix(scipy.linalg.solve_continuous_are(a, b, q, r))
    # compute the LQR gain
    k = np.array(scipy.linalg.inv(r) * (b.T * x))

    eig_vals, eig_vecs = scipy.linalg.eig(a - b * k)

    return k, x, eig_vals


@dataclass
class LQRParam:
    r: float = 1
    """ Input Multiplier """
    q: np.ndarray = np.identity(2)
    """State Multiplier """


class LQR:

    def __init__(self, params: LQRParam = LQRParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.path: Optional[DgLanelet] = None
        self.u: np.ndarray
        self.params: LQRParam = params
        self.vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()
        # logger.debug("Pure pursuit params: \n", self.param)

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_state(self, obs: X):
        self.pose = SE2_from_translation_angle([obs.x, obs.y], obs.theta)
        lanepose = self.path.lane_pose_from_SE2_generic(self.pose)
        error = np.array([[lanepose.lateral], [lanepose.relative_heading]])
        speed = obs.vx

        beta, _ = self.path.find_along_lane_closest_point(np.array([obs.x, obs.y]))
        beta1, beta2 = beta-1, beta+1
        q1, q2 = self.path.center_point(beta1), self.path.center_point(beta2)
        trans1, angle1 = translation_angle_from_SE2(q1)
        trans2, angle2 = translation_angle_from_SE2(q2)
        delta_angle = angle2 - angle1
        delta_pos = np.linalg.norm(trans2 - trans1)
        feed_forward = math.atan(delta_angle/delta_pos * self.vehicle_geometry.length)

        a = np.array([[0, speed], [0, 0]])
        b = np.array([[0], [speed/self.vehicle_geometry.length]])
        k, _, _ = lqr(a, b, self.params.q, np.array([[self.params.r]]))
        self.u = - np.matmul(k, error) + feed_forward

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.pose, self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")

        return self.u
