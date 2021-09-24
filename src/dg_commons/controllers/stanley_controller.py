from dataclasses import dataclass
from math import sin, atan
from typing import Optional, Tuple
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleState

import numpy as np
import scipy.optimize
from geometry import SE2value, translation_angle_from_SE2, SE2_from_rotation_translation, SE2_from_translation_angle
from geometry.rotations import SO2_from_angle

from dg_commons.geo import euclidean_between_SE2value
from dg_commons.planning.lanes import DgLanelet
from games import X, U


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
        pose = SE2_from_translation_angle([obs.x, obs.y], obs.theta)
        tr, ang = translation_angle_from_SE2(pose)
        rot = SO2_from_angle(ang)

        delta_tr = np.dot(rot, np.array([self.vehicle_geometry.lf, 0]).T)
        tr += delta_tr
        self.front_pose = SE2_from_rotation_translation(rot, tr)

        lanepose = self.path.lane_pose_from_SE2_generic(self.front_pose)
        if X == VehicleStateDyn:
            front_speed = np.array(obs.vx, obs.vy) + obs.dtheta*np.array(0, self.vehicle_geometry.lf)
            self.front_speed = np.linalg.norm(front_speed)
        else:
            self.front_speed = obs.vx/np.cos(obs.delta)

        self.alpha = -lanepose.relative_heading
        self.lateral = -lanepose.lateral

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.alpha, self.lateral, self.front_speed]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.alpha + atan(self.params.stanley_gain*self.lateral/self.front_speed)
