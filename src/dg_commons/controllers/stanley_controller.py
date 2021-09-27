from dataclasses import dataclass
from math import atan
from typing import Optional
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_dynamic import VehicleStateDyn
import numpy as np
from geometry import SE2value, SE2_from_translation_angle, translation_angle_scale_from_E2, translation_angle_from_SE2
from dg_commons.planning.lanes import DgLanelet
from games import X
from duckietown_world.utils import SE2_apply_R2
import math
from duckietown_world import relative_pose
from dg_commons.controllers.mpc.mpc_utils.path_approximation_techniques import linear_param


__all__ = ["Stanley", "StanleyParam"]


@dataclass
class StanleyParam:
    stanley_gain: float = 1
    """ Tunable gain """


class Stanley:
    """
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/amod
    /AMOD_2020/20201019-05%20-%20ETHZ%20-%20Control%20in%20Duckietown%20(PID).pdf
    Note there is an error in computation of alpha (order needs to be inverted)
    """

    def __init__(self, params: StanleyParam = StanleyParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.path: Optional[DgLanelet] = None
        self.front_pose: Optional[SE2value] = None
        self.speed: Optional[float] = None
        self.alpha: Optional[float] = None
        self.lateral: Optional[float] = None
        self.params: StanleyParam = params
        self.vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()
        # logger.debug("Pure pursuit params: \n", self.param)

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_state(self, obs: X):
        tr, ang = [obs.x, obs.y], obs.theta
        pose = SE2_from_translation_angle(tr, ang)

        front_position = SE2_apply_R2(pose, np.array([self.vehicle_geometry.lf, 0]))
        front_pose = SE2_from_translation_angle(front_position, ang)

        front_speed = False
        if front_speed:
            if X == VehicleStateDyn:
                front_speed = np.array(obs.vx, obs.vy) + obs.dtheta*np.array(0, self.vehicle_geometry.lf)
                self.speed = np.linalg.norm(front_speed)
            else:
                self.speed = obs.vx/math.cos(obs.delta)
        else:
            self.speed = obs.vx

        p, _, _ = translation_angle_scale_from_E2(front_pose)
        beta, q0 = self.path.find_along_lane_closest_point(p, tol=1e-4)

        path_approx = True
        if path_approx:
            pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(beta)
            res, func, vertical_line = linear_param(pos1, angle1, pos2, angle2, pos3, angle3)
            angle = res[2]
            self.alpha = angle - obs.theta
            if vertical_line:
                self.lateral = res[0] - front_position[0]
            else:
                x_val = (front_position[0] + res[0] * (front_position[1] - res[1])) / (1 + res[0] ** 2)
                y_val = (res[0] ** 2 * front_position[1] + res[0] * front_position[0] + res[1]) / (1 + res[0] ** 2)
                self.lateral = - (x_val - front_position[0])*math.sin(obs.theta) + \
                                 (y_val - front_position[1])*math.cos(obs.theta)
        else:
            rel = relative_pose(front_pose, q0)

            r, self.alpha, _ = translation_angle_scale_from_E2(rel)
            self.lateral = r[1]

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
        if any([_ is None for _ in [self.alpha, self.lateral, self.speed]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")

        return self.alpha + atan(self.params.stanley_gain*self.lateral/self.speed)
