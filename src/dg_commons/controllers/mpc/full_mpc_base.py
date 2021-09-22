from dataclasses import dataclass
from typing import Tuple
from abc import abstractmethod
from dg_commons.controllers.mpc.lateral_mpc_base import VEHICLE_PARAMS, LatMPCKinBaseAnalytical, \
    LatMPCKinBasePathVariable, LatMPCKinBaseParam


@dataclass
class FullMPCKinBaseParam(LatMPCKinBaseParam):
    velocity_err_weight: float = 1
    """ Weighting factor in cost function for velocity error """
    acceleration_weight: float = 1
    """ Weighting factor in cost function for acceleration """
    acc_bounds: Tuple[float, float] = VEHICLE_PARAMS.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = VEHICLE_PARAMS.vx_limits


class FullMPCKinBaseAnalytical(LatMPCKinBaseAnalytical):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.a = self.model.set_variable(var_type='_u', var_name='a')

    def set_bounds(self):
        super().set_bounds()
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]


class FullMPCKinBasePathVariable(LatMPCKinBasePathVariable):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.a = self.model.set_variable(var_type='_u', var_name='a')

    def set_bounds(self):
        super().set_bounds()
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]
