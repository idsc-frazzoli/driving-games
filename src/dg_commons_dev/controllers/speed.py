from dataclasses import dataclass
from math import pi
from typing import MutableMapping, Dict, Tuple, Optional, List, Union
import numpy as np
from duckietown_world import relative_pose, SE2Transform
from geometry import SE2value
from dg_commons import PlayerName
from dg_commons.controllers.pid import PIDParam, PID
from games.utils import valmap
from dg_commons.sim.models import extract_pose_from_state, kmh2ms, extract_vel_from_state
from dg_commons.sim.simulator_structures import PlayerObservations
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.controllers.controller_types import LongitudinalControllerParam, LongitudinalController
from dg_commons import X

__all__ = ["SpeedControllerParam", "SpeedController"]


@dataclass
class SpeedControllerParam(LongitudinalControllerParam, PIDParam):
    """Default values are tuned roughly for a default car model"""
    kP: Union[float, List[float]] = 4
    kI: Union[float, List[float]] = 0.01
    kD: Union[float, List[float]] = 0.1
    antiwindup: Union[Tuple[float, float], List[Tuple[float, float]]] = (-2, 2)
    setpoint_minmax: Union[Tuple[float, float], List[Tuple[float, float]]] = (-kmh2ms(10), kmh2ms(150))
    output_minmax: Union[Tuple[float, float], List[Tuple[float, float]]] = (-8, 5)  # acc minmax

    def __post_init__(self):
        if isinstance(self.antiwindup, list):
            assert all([i[0] < i[1] for i in self.antiwindup])
        else:
            assert self.antiwindup[0] < self.antiwindup[1]
        if isinstance(self.setpoint_minmax, list):
            assert all([i[0] < i[1] for i in self.setpoint_minmax])
        else:
            assert self.setpoint_minmax[0] < self.setpoint_minmax[1]
        if isinstance(self.output_minmax, list):
            assert all([i[0] < i[1] for i in self.output_minmax])
        else:
            assert self.output_minmax[0] < self.output_minmax[1]
        super().__post_init__()


class SpeedController(PID, LongitudinalController):
    """Low-level controller for reference tracking of speed"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SpeedControllerParam() if params is None else params
        super(SpeedController, self).__init__(params)

    def update_reference_speed(self, speed_ref: float):
        self.update_reference(speed_ref)

    def update_state(self, state: X):
        self.update_measurement(state.vx)

    def get_acceleration(self, at: float) -> float:
        return self.get_control(at)
