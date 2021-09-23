from typing import Optional
from sim.models.vehicle_structures import VehicleGeometry
import scipy.optimize
from dg_commons.planning.lanes import DgLanelet
from games import X, U
import scipy.linalg
from dataclasses import dataclass
import numpy as np
from duckietown_world import relative_pose
from duckietown_world.utils import  SE2_apply_R2
from geometry import  SE2_from_translation_angle,  translation_angle_from_SE2
from dg_commons.utils import SemiDef


__all__ = ["LQR", "LQRParam"]


def lqr(a, b, q, r):
    """Solve the continuous time lqr controller.

    dx/dt = ax + bu

    cost = integral x.T*q*x + u.T*r*u
    """
    # first, try to solve the ricatti equation
    x = np.matrix(scipy.linalg.solve_continuous_are(a, b, q.matrix, r.matrix))
    # compute the LQR gain
    k = np.array(scipy.linalg.inv(r.matrix) * (b.T * x))

    eig_vals, eig_vecs = scipy.linalg.eig(a - b * k)

    return k, x, eig_vals


@dataclass
class LQRParam:
    r: SemiDef = SemiDef([1])
    """ Input Multiplier """
    q: SemiDef = SemiDef(matrix=np.identity(3))
    """State Multiplier """
    t_step: float = 0.1


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
        self.pose = None
        # logger.debug("Pure pursuit params: \n", self.param)

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_state(self, obs: X):
        pose = SE2_from_translation_angle(np.array([obs.x, obs.y]), obs.theta)

        position = SE2_apply_R2(pose, np.array([-self.vehicle_geometry.lr, 0]))
        angle = obs.theta
        self.pose = SE2_from_translation_angle(position, angle)

        beta, q0 = self.path.find_along_lane_closest_point(position, tol=1e-4)
        rel = relative_pose(q0, self.pose)

        r, relative_heading = translation_angle_from_SE2(rel)
        lateral = r[1]
        error = np.array([[lateral], [relative_heading], [obs.delta]])

        speed = obs.vx

        feed_forward = 0

        a = np.array([[0, speed, 0], [0, 0, speed/self.vehicle_geometry.length], [0, 0, 0]])
        b = np.array([[0], [0], [1]])
        k, _, _ = lqr(a, b, self.params.q, self.params.r)
        self.u = -np.matmul(k, error) + feed_forward

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.pose, self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")

        return self.u
