from typing import Tuple
from abc import abstractmethod
from dg_commons_dev.controllers.mpc.lateral_mpc_base import vehicle_params, LatMPCKinBase, LatMPCKinBaseParam
from dg_commons_dev.controllers.mpc.mpc_utils.cost_functions import *


@dataclass
class FullMPCKinBaseParam(LatMPCKinBaseParam):
    cost: str = "quadratic"
    """ Cost function """
    cost_params: CostParameters = QuadraticParams(
        q=SemiDef(matrix=np.eye(3)),
        r=SemiDef(matrix=np.eye(2))
    )
    """ Cost function parameters """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits


class FullMPCKinBase(LatMPCKinBase):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.a = self.model.set_variable(var_type='_u', var_name='a')

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        lterm, _ = costs[self.params.cost](error, inp, self.params.cost_params)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        _, mterm = costs[self.params.cost](error, inp, self.params.cost_params)
        return mterm

    def set_bounds(self):
        super().set_bounds()
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]
