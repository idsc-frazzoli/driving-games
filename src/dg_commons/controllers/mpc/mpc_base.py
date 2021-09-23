from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from games import X
import do_mpc
from dg_commons.controllers.mpc.mpc_utils import *


@dataclass
class MPCKinBAseParam:
    n_horizon: int = 15
    """ Horizon Length """
    t_step: float = 0.1
    """ Sample Time """
    cost: str = None
    """ Cost function """
    cost_params: CostParameters = None
    """ Cost function parameters """


class MPCKinBase(ABC):
    @abstractmethod
    def __init__(self, params, model_type: str):
        self.params = params
        self.mpc = None
        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }

        self.model = do_mpc.model.Model(model_type)
        self.state_x = self.model.set_variable(var_type='_x', var_name='state_x', shape=(1, 1))
        self.state_y = self.model.set_variable(var_type='_x', var_name='state_y', shape=(1, 1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        self.v = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        self.delta = self.model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))

    @abstractmethod
    def update_state(self, obs: Optional[X] = None, speed_ref: Optional[float] = None):
        pass

    @abstractmethod
    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def set_scaling(self):
        pass
