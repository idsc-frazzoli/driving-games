from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class BaseParam:
    pass


class BaseClass(ABC):
    @abstractmethod
    def __init__(self, base_param: BaseParam):
        pass

    @abstractmethod
    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return desired_steering


@dataclass
class SCIdentityParam(BaseParam):
    pass


class SCIdentity(BaseClass):
    def __init__(self, params: SCIdentityParam):
        self.params = SCIdentityParam() if params is None else params

    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return desired_steering


@dataclass
class SCPParam(BaseParam):
    ddelta_kp: float = 10


class SCP(BaseClass):
    def __init__(self, params: Optional[SCPParam] = None):
        self.params = SCPParam() if params is None else params

    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return self.params.ddelta_kp*(desired_steering - current_steering)
