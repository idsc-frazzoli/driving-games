from dataclasses import dataclass
from typing import Optional, Union, List
from abc import ABC, abstractmethod
from dg_commons_dev.utils import BaseParams


class BaseClass(ABC):
    @abstractmethod
    def __init__(self, base_param: BaseParams):
        pass

    @abstractmethod
    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return desired_steering


@dataclass
class SCIdentityParam(BaseParams):
    pass


class SCIdentity(BaseClass):
    def __init__(self, params: Optional[SCIdentityParam] = None):
        self.params = SCIdentityParam() if params is None else params

    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return desired_steering


@dataclass
class SCPParam(BaseParams):
    ddelta_kp: Union[List[float], float] = 10


class SCP(BaseClass):
    def __init__(self, params: Optional[SCPParam] = None):
        self.params = SCPParam() if params is None else params

    def get_steering_velocity(self, desired_steering: float, current_steering: float):
        return self.params.ddelta_kp*(desired_steering - current_steering)
