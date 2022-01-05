from dataclasses import dataclass
from typing import Tuple
import do_mpc
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState

from dg_commons import X
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from homotopies.utils import *

vehicle_params = VehicleParameters.default_car()

@dataclass
class MpcKinBaseParams:
    n_horizon: int = 15
    """ Horizon Length """
    t_step: float = 0.1
    """ Sample Time """
    cost = QuadraticCost
    """ Cost function """
    cost_params: Union[List[CostParameters], CostParameters] = QuadraticParams(
        q=SemiDef(eig=[1, 1, 1]),
        r=SemiDef(matrix=np.eye(2))
    )
    """ Cost function parameters """
    delta_input_weight: float = 1e-2
    """ Weighting factor in cost function for varying input """
    vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()

    v_delta_bounds: Tuple[float, float] = (-vehicle_params.ddelta_max, vehicle_params.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Tuple[float, float] = (-vehicle_params.default_car().delta_max, vehicle_params.default_car().delta_max)
    """ Steering Bounds """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits #vx in model frame


class MpcKinBase:
    def __init__(self, params, model_type: str):
        self.params = params
        self.model = do_mpc.model.Model(model_type)

        self.state_x = self.model.set_variable(var_type="_x", var_name="state_x", shape=(1, 1))
        self.state_y = self.model.set_variable(var_type="_x", var_name="state_y", shape=(1, 1))
        self.theta = self.model.set_variable(var_type="_x", var_name="theta", shape=(1, 1))
        self.v = self.model.set_variable(var_type="_x", var_name="v", shape=(1, 1))
        self.delta = self.model.set_variable(var_type="_x", var_name="delta", shape=(1, 1))

        self.a = self.model.set_variable(var_type="_u", var_name="a")
        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')

        self.obstacle_obs: VehicleState = None
        self.obstacle_obs_flag = False
        self.obstacle_state = self.model.set_variable(var_type='_tvp', var_name='obstacle_state', shape=(5, 1))

        self.homotopy = self.model.set_variable('_p', 'homotopy')#0 for overtaking from left, 1 for overtaking from right

        self.cost = self.params.cost(self.params.cost_params)

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }
