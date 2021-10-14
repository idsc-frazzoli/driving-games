from dataclasses import dataclass
from typing import Optional, Union, List
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.controllers.controller_types import SteeringController, SteeringControllerParam


@dataclass
class SCIdentityParam(SteeringControllerParam):
    pass


class SCIdentity(SteeringController):
    def __init__(self, params: Optional[SCIdentityParam] = None):
        self.params = SCIdentityParam() if params is None else params

    def get_steering_vel(self, desired_steering: float, current_steering: float) -> float:
        return desired_steering


@dataclass
class SCPParam(SteeringControllerParam):
    ddelta_kp: Union[List[float], float] = 10


class SCP(SteeringController):
    def __init__(self, params: Optional[SCPParam] = None):
        self.params = SCPParam() if params is None else params

    def get_steering_vel(self, desired_steering: float, current_steering: float) -> float:
        return self.params.ddelta_kp * (desired_steering - current_steering)
