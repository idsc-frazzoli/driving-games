from math import pi
from typing import Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from aido_schemas import DTSimRobotInfo
from duckietown_world import relative_pose, SE2Transform
from geometry import xytheta_from_SE2

__all__ = ["SpeedController", "SpeedBehavior"]


@dataclass
class SpeedControllerParam:
    kP: float = 0.1
    kI: float = 0.0
    antiwindup: Tuple[float, float] = (-0.5, 0.5)
    setpoint_minmax: Tuple[float, float] = (-1, 1)


class SpeedController:
    """reference tracking of speed"""

    def __init__(self):
        self.params = SpeedControllerParam()
        self.current_speed: float = 0
        self.desired_speed: float = 0
        self.last_request_at: Optional[float] = None
        self.last_integral_error: float = 0

    def update_observations(self, current_velocity: np.ndarray):
        self.current_speed = xytheta_from_SE2(current_velocity)[0]

    def update_reference(self, desired_speed: float):
        if not self.params.setpoint_minmax[0] <= desired_speed <= self.params.setpoint_minmax[1]:
            raise RuntimeWarning("Attempting to set a desired speed out of range. I'll clip the value.")
        self.desired_speed = np.clip(desired_speed, self.params.setpoint_minmax[0], self.params.setpoint_minmax[1])

    def get_control(self, at: float) -> float:
        "A simple PI"
        dt = 0 if self.last_request_at is None else at - self.last_request_at
        self.last_request_at = at
        p_error = self.desired_speed - self.current_speed
        self.last_integral_error += self.params.kI * p_error * dt
        self.last_integral_error = np.clip(self.last_integral_error,
                                           self.params.antiwindup[0],
                                           self.params.antiwindup[1])
        return self.params.kP * p_error + self.last_integral_error


@dataclass
class SpeedBehaviorParam:
    nominal_speed: float = 0.2
    safety_dist_right: float = 0.7
    safety_dist_front: float = 0.25


class SpeedBehavior:
    """Determines the reference speed"""
    duckiebots: Dict[str, DTSimRobotInfo]
    last_speed_ref: float = 0

    def __init__(self, myname: str):
        self.params: SpeedBehaviorParam = SpeedBehaviorParam()
        self.myname: str = myname

    def update_observations(self, duckiebots: Dict[str, DTSimRobotInfo]):
        self.duckiebots = duckiebots

    def get_speed_ref(self, at: float) -> float:
        """Check if there is anyone on the right too close, then brake"""

        yield_to_anyone: bool = self.is_there_anyone_to_yield_to()
        if yield_to_anyone:
            self.last_speed_ref = 0
        else:
            self.last_speed_ref = self.params.nominal_speed
        return self.last_speed_ref

    def is_there_anyone_to_yield_to(self) -> bool:
        """
        If someone is approaching from the right or someone is in front of us we yield
        """

        for dk_name, dk_sim_robot in self.duckiebots.items():
            if dk_name == self.myname:
                pass
            rel = SE2Transform.from_SE2(relative_pose(
                self.duckiebots[self.myname].pose, self.duckiebots[dk_name].pose))

            distance = np.linalg.norm(rel.p)
            coming_from_the_right: bool = pi / 4 <= rel.theta <= pi * 3 / 4
            in_front_of_me: bool = rel.p[0] > 0 and -pi / 4 <= rel.theta <= pi / 4
            if (coming_from_the_right and distance < self.params.safety_dist_right) or (
                    in_front_of_me and distance < self.params.safety_dist_front):
                return True
        return False