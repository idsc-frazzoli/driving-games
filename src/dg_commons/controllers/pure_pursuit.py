from dataclasses import dataclass
from math import sin
from typing import Optional, Tuple

import numpy as np
import scipy.optimize
from duckietown_world import LaneSegment
from geometry import SE2value, translation_angle_from_SE2
from gtduckie.utils import euclidean_between_SE2value

__all__ = ["PurePursuit"]


@dataclass
class PurePursuitParam:
    look_ahead: float = 0.25
    min_distance: float = 0.05
    max_extra_distance: float = 0.4
    k_turn2pwm: float = 0.3


class PurePursuit:
    """
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/amod
    /AMOD_2020/20201019-05%20-%20ETHZ%20-%20Control%20in%20Duckietown%20(PID).pdf
    """

    def __init__(self, params: PurePursuitParam = PurePursuitParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.path: Optional[LaneSegment] = None
        self.rel_pose: Optional[SE2value] = None
        self.along_path: Optional[float] = None
        self.speed: float = 0
        self.param: PurePursuitParam = params
        print("Pure pursuit params: \n", self.param)

    def update_path(self, path: LaneSegment):
        assert isinstance(path, LaneSegment)
        self.path = path

    def update_pose(self, rel_pose: SE2value, along_path: float):
        assert isinstance(rel_pose, SE2value)
        assert isinstance(along_path, float)
        self.rel_pose = rel_pose
        self.along_path = along_path

    def update_speed(self, speed: float):
        self.speed = speed

    def find_goal_point(self) -> Tuple[float, SE2value]:
        """
        Find goal point along the path
        :return: along_path, SE2value
        """

        def goal_point_error(along_path: float) -> float:
            """
            :param along_path:
            :return: Error between desired distance from pose to point along path
            """
            beta = self.path.beta_from_along_lane(along_path)
            cp = self.path.center_point(beta)
            dist = euclidean_between_SE2value(self.rel_pose, cp)
            return np.linalg.norm(dist - self.param.look_ahead)

        min_along_path = self.along_path + self.param.min_distance

        bounds = [min_along_path,
                  min_along_path + self.param.look_ahead + self.param.max_extra_distance]
        res = scipy.optimize.minimize_scalar(fun=goal_point_error, bounds=bounds, method='Bounded')
        goal_point = self.path.center_point(self.path.beta_from_along_lane(res.x))
        return res.x, goal_point

    def get_steering_derivative(self) -> float:
        """
        gives "rotational velocity"
        :return: float
        """
        # todo fixme
        if any([_ is None for _ in [self.rel_pose, self.path]]):
            raise RuntimeError("Attempting to use pure pursuit before having set any observations/path")
        p, theta = translation_angle_from_SE2(self.rel_pose)
        _, goal_point = self.find_goal_point()
        p_goal, theta_goal = translation_angle_from_SE2(goal_point)
        alpha = theta - np.arctan2(p_goal[1] - p[1], p_goal[0] - p[0])
        radius = self.param.look_ahead / (2 * sin(alpha))
        return - self.param.k_turn2pwm * self.speed / radius
