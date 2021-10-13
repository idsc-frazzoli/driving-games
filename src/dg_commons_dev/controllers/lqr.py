from typing import Optional, List, Union
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
import scipy.optimize
from dg_commons.maps.lanes import DgLanelet
from dg_commons import X
import scipy.linalg
from dataclasses import dataclass
import numpy as np
from duckietown_world import relative_pose
from duckietown_world.utils import SE2_apply_R2
from geometry import SE2_from_translation_angle,  translation_angle_from_SE2, translation_angle_scale_from_E2
from dg_commons_dev.utils import SemiDef
import math
from dg_commons_dev.controllers.path_approximation_techniques import PathApproximationTechniques, LinearPath
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.maps.lanes import DgLaneletControl


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
class LQRParam(BaseParams):
    r: Union[List[SemiDef], SemiDef] = SemiDef([1])
    """ Input Multiplier """
    q: Union[List[SemiDef], SemiDef] = SemiDef(matrix=np.identity(3))
    """State Multiplier """
    t_step: Union[List[float], float] = 0.1


class LQR:

    USE_STEERING_VELOCITY: bool = True

    def __init__(self, params: LQRParam = LQRParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.path: Optional[DgLanelet] = None
        self.control_path: Optional[DgLaneletControl] = None
        self.u: np.ndarray
        self.params: LQRParam = params
        self.vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()
        self.back_pose = None
        self.speed: Optional[float] = None
        self.current_beta = None
        self.path_approx = LinearPath()

        # logger.debug("Pure pursuit params: \n", self.param)

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path
        self.control_path = DgLaneletControl(path)

    def update_state(self, obs: X):
        pose = SE2_from_translation_angle(np.array([obs.x, obs.y]), obs.theta)

        back_position = SE2_apply_R2(pose, np.array([-self.vehicle_geometry.lr, 0]))
        angle = obs.theta
        self.back_pose = SE2_from_translation_angle(back_position, angle)
        self.speed = obs.vx

        p, _, _ = translation_angle_scale_from_E2(self.back_pose)

        control_sol_params = self.control_path.ControlSolParams(obs.vx, self.params.t_step)
        self.current_beta, q0 = self.control_path.find_along_lane_closest_point(back_position, tol=1e-4,
                                                                                control_sol=control_sol_params)

        path_approx = True
        if path_approx:
            pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
            self.path_approx.update_from_data(pos1, angle1, pos2, angle2, pos3, angle3)
            res = self.path_approx.parameters
            closest_point_func = self.path_approx.closest_point_on_path

            angle = res[2]
            relative_heading = - angle + obs.theta
            if relative_heading > math.pi:
                relative_heading = -(2*math.pi - relative_heading)
            elif relative_heading < - math.pi:
                relative_heading = 2*math.pi + relative_heading

            closest_point = closest_point_func(back_position)
            lateral = (closest_point[0] - back_position[0]) * math.sin(obs.theta) - \
                      (closest_point[1] - back_position[1]) * math.cos(obs.theta)
        else:
            rel = relative_pose(q0, self.back_pose)
            r, relative_heading = translation_angle_from_SE2(rel)
            lateral = r[1]

        error = np.array([[lateral], [relative_heading], [obs.delta]])

        feed_forward = 0

        a = np.array([[0, self.speed, 0], [0, 0, self.speed/self.vehicle_geometry.length], [0, 0, 0]])
        b = np.array([[0], [0], [1]])
        try:
            k, _, _ = lqr(a, b, self.params.q, self.params.r)
            self.u = -np.matmul(k, error) + feed_forward
        except np.linalg.LinAlgError:
            self.u = 0

    def next_pos(self, current_beta):
        along_lane = self.path.along_lane_from_beta(current_beta)
        k = 2
        delta_step = self.speed * 0.1 * k
        along_lane1 = along_lane + delta_step / 2
        along_lane2 = along_lane1 + delta_step / 2

        beta1, beta2, beta3 = current_beta, self.path.beta_from_along_lane(along_lane1), \
                              self.path.beta_from_along_lane(along_lane2)

        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)
        q3 = self.path.center_point(beta3)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)
        pos3, angle3 = translation_angle_from_SE2(q3)
        self.target_position = pos3
        return pos1, angle1, pos2, angle2, pos3, angle3

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.back_pose, self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")

        return self.u
