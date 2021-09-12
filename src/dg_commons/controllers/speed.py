from dataclasses import dataclass
from math import pi
from typing import Optional, Tuple, MutableMapping

import numpy as np
from duckietown_world import relative_pose, SE2Transform

from dg_commons.controllers.pid import PIDParam, PID
from games import PlayerName, X

__all__ = ["SpeedControllerParam", "SpeedController", "SpeedBehavior"]

from sim.models import extract_pose_from_state, kmh2ms, extract_vel_from_state


@dataclass
class SpeedControllerParam(PIDParam):
    """Default values are tuned roughly for a default car model"""
    kP: float = 4
    kI: float = 0.01
    kD: float = 0.1
    antiwindup: Tuple[float, float] = (-2, 2)
    setpoint_minmax: Tuple[float, float] = (-kmh2ms(10), kmh2ms(150))
    output_minmax: Tuple[float, float] = (-8, 5)  # acc minmax


class SpeedController(PID):
    """Low-level controller for reference tracking of speed"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SpeedControllerParam() if params is None else params
        super(SpeedController, self).__init__(params)


@dataclass
class SpeedBehaviorParam:
    nominal_speed: float = kmh2ms(40)
    safety_dist_lateral: float = 7
    safety_dist_front: float = 6
    safety_dist_front_crash: float = 10
    safety_time_front: float = 2
    minimum_safety_vel: float = kmh2ms(5)


class SpeedBehavior:
    """Determines the reference speed"""

    def __init__(self, my_name: Optional[PlayerName] = None):
        self.params: SpeedBehaviorParam = SpeedBehaviorParam()
        self.my_name: PlayerName = my_name
        self.agents: Optional[MutableMapping[PlayerName, X]] = None
        self.speed_ref: float = 0
        """ The speed reference"""

    def update_observations(self, agents: MutableMapping[PlayerName, X]):
        self.agents = agents

    def get_speed_ref(self, at: float) -> float:
        """Check if there is anyone on the right too close, then brake"""

        yield_to_anyone: bool = self.is_there_anyone_to_yield_to()
        if yield_to_anyone:
            self.speed_ref = 0
        else:
            self.speed_ref = self.cruise_control()
        return self.speed_ref

    def is_there_anyone_to_yield_to(self) -> bool:
        """
        If someone is approaching from the right or someone is in front of us we yield
        """

        mypose = extract_pose_from_state(self.agents[self.my_name])
        for other_name, _ in self.agents.items():
            if not other_name == self.my_name:
                rel = SE2Transform.from_SE2(relative_pose(
                    mypose, extract_pose_from_state(self.agents[other_name])))
                vel = extract_vel_from_state(self.agents[other_name])

                distance = np.linalg.norm(rel.p)
                coming_from_the_right: bool = pi / 4 <= rel.theta <= pi * 3 / 4 and \
                                              vel > self.params.minimum_safety_vel
                coming_from_the_left: bool = -3 * pi / 4 <= rel.theta <= -pi / 4 and \
                                             vel > self.params.minimum_safety_vel
                in_front_of_me: bool = rel.p[0] > 0 and - 1.2 <= rel.p[1] <= 1.2
                coming_from_the_front: bool = 3 * pi / 4 <= abs(rel.theta) <= pi * 5 / 4 and in_front_of_me
                if (coming_from_the_right and distance < self.params.safety_dist_lateral) or (
                        coming_from_the_left and distance < self.params.safety_dist_lateral) or (
                        coming_from_the_front and distance < self.params.safety_dist_front_crash):
                    return True
        return False

    def cruise_control(self) -> float:
        """
        If someone is in front with the same orientation, then apply the two seconds rule to adapt reference velocity
         that allows maintaining a safe distance between the vehicles
        """
        mypose = extract_pose_from_state(self.agents[self.my_name])
        myvel = self.agents[self.my_name].vx
        for other_name, _ in self.agents.items():
            if not other_name == self.my_name:
                rel = SE2Transform.from_SE2(relative_pose(
                    mypose, extract_pose_from_state(self.agents[other_name])))
                rel_dist = np.linalg.norm(rel.p)
                vel = self.agents[other_name].vx
                in_front_of_me: bool = rel.p[0] > 0 and - 1.2 <= rel.p[1] <= 1.2
                if in_front_of_me and rel_dist < self.params.safety_dist_front + self.params.safety_time_front * \
                        abs(vel - myvel):
                    return float(np.clip(vel - abs(vel - myvel), 0, kmh2ms(200)))
        return self.params.nominal_speed
