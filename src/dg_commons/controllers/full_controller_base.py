from dg_commons.controllers.controller_types import *
from dg_commons.state_estimators.estimator_types import *
from sim.agents.lane_followers import LaneFollowerAgent
from dataclasses import dataclass
from typing import Union, Optional, get_args
import os


@dataclass
class VehicleController:

    controller: type(Union[LateralController, LatAndLonController])

    controller_params: Union[LateralControllerParam, LatAndLonControllerParam]

    lf_agent: type(LaneFollowerAgent)

    speed_behavior_param: SpeedBehaviorParam

    steering_controller: type(SteeringController) = SCIdentity

    steering_controller_params: SteeringControllerParam = SCIdentityParam()

    longitudinal_controller: Optional[type(LongitudinalController)] = None

    longitudinal_controller_params: Optional[LongitudinalControllerParam] = None

    state_estimator: Optional[type(Estimators)] = None

    state_estimator_params: Optional[EstimatorsParams] = None

    _extra_folder_name: str = ""

    def __post_init__(self):
        self._extra_folder_name = self.extra_folder_name
        self.folder_name = os.path.join(self.controller.__name__, self._extra_folder_name)

        decoupled: bool = self.controller in get_args(LateralController) and \
                          self.longitudinal_controller in get_args(LongitudinalController)

        single: bool = self.controller in get_args(LatAndLonController) and \
                       self.longitudinal_controller is None

        assert single or decoupled
        self.decoupled = decoupled

        if self.steering_controller is not SCIdentity:
            assert type(self.steering_controller_params) is not SCIdentityParam

        if self.longitudinal_controller is not None:
            assert self.longitudinal_controller_params is not None

        if self.state_estimator is not None:
            assert self.state_estimator_params is not None

    @property
    def extra_folder_name(self):
        return self._extra_folder_name

    @extra_folder_name.setter
    def extra_folder_name(self, name):
        self._extra_folder_name = name
        self.folder_name = os.path.join(self.controller.__name__, name)
