from dataclasses import dataclass
from math import atan
from typing import Optional
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_dynamic import VehicleStateDyn
import numpy as np
from geometry import SE2value, SE2_from_translation_angle, translation_angle_scale_from_E2
from dg_commons.planning.lanes import DgLanelet
from games import X
from duckietown_world.utils import SE2_apply_R2
import math
from duckietown_world import relative_pose


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
        self.front_speed: Optional[float] = None
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

        lanepose = self.path.lane_pose_from_SE2_generic(front_pose)
        if X == VehicleStateDyn:
            front_speed = np.array(obs.vx, obs.vy) + obs.dtheta*np.array(0, self.vehicle_geometry.lf)
            self.front_speed = np.linalg.norm(front_speed)
        else:
            self.front_speed = obs.vx/math.cos(obs.delta)

        self.alpha = -lanepose.relative_heading
        self.lateral = -lanepose.lateral

        alternative = True
        if alternative:
            self.lateral, self.alpha = self.alternative(front_pose)

    def alternative(self, front_pose):
        p, _, _ = translation_angle_scale_from_E2(front_pose)

        beta, q0 = self.path.find_along_lane_closest_point(p, tol=1e-4)
        rel = relative_pose(front_pose, q0)

        r, relative_heading, _ = translation_angle_scale_from_E2(rel)
        lateral = r[1]
        return lateral, relative_heading

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.alpha, self.lateral, self.front_speed]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")

        return self.alpha + atan(self.params.stanley_gain*self.lateral/self.front_speed)
