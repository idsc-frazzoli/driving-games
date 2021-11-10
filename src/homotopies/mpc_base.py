from dataclasses import dataclass
from typing import Tuple
import do_mpc
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState

from dg_commons import X
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons_dev.controllers.utils.cost_functions import *
from dg_commons_dev.curve_approximation_techniques import *
from dg_commons_dev.controllers.controller_types import LatAndLonController

vehicle_params = VehicleParameters.default_car()

@dataclass
class MpcKinBaseParams:
    n_horizon: Union[List[int], int] = 15
    """ Horizon Length """
    t_step: Union[List[float], float] = 0.1
    """ Sample Time """
    cost: Union[List[CostFunctions], CostFunctions] = QuadraticCost
    """ Cost function """
    cost_params: Union[List[CostParameters], CostParameters] = QuadraticParams(
        q=SemiDef(matrix=np.eye(3)),
        r=SemiDef(matrix=np.eye(2))
    )
    """ Cost function parameters """
    delta_input_weight: Union[List[float], float] = 1e-2
    """ Weighting factor in cost function for varying input """
    vehicle_geometry: Union[List[VehicleGeometry], VehicleGeometry] = VehicleGeometry.default_car()

    v_delta_bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (-vehicle_params.ddelta_max,
                                                                             vehicle_params.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (-vehicle_params.default_car().delta_max,
                                                                           vehicle_params.default_car().delta_max)
    """ Steering Bounds """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits

    def __post_init__(self):
        if isinstance(self.cost, list):
            assert len(self.cost) == len(self.cost_params)
            for i, technique in enumerate(self.cost):
                assert MapCostParam[technique] == type(self.cost_params[i])

        else:
            assert MapCostParam[self.cost] == type(self.cost_params)


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

        self.homotopy_classes = self.model.set_variable('_p', 'homotopy')#0 for overtaking from left, 1 for overtaking from right

        self.cost = self.params.cost(self.params.cost_params)

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }


