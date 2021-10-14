from typing import Union, Optional
from dg_commons.maps.lanes import DgLanelet
from dg_commons_dev.maps.lanes import DgLaneletControl
from abc import ABC, abstractmethod
from dg_commons import X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


@dataclass
class LateralControllerParam(BaseParams):
    pass


class LateralController(ABC):
    def __init__(self):
        self.path: Optional[DgLanelet] = None
        self.control_path: Optional[DgLaneletControl] = None

    @abstractmethod
    def get_steering(self) -> float:
        pass

    @abstractmethod
    def update_state(self, state: X):
        pass

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path
        self.control_path = DgLaneletControl(path)


@dataclass
class LongitudinalControllerParam(BaseParams):
    pass


class LongitudinalController(ABC):
    def __init__(self):
        self.speed_ref: float = 0

    @abstractmethod
    def get_acceleration(self, at: float) -> float:
        pass

    @abstractmethod
    def update_state(self, state: X):
        pass

    def update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref


@dataclass
class LatAndLonControllerParam(BaseParams):
    pass


class LatAndLonController(LateralController, LongitudinalController, ABC):
    pass


@dataclass
class SteeringControllerParam(BaseParams):
    pass


class SteeringController(ABC):
    @abstractmethod
    def get_steering_vel(self, desired_steering: float, current_steering: float) -> float:
        pass
