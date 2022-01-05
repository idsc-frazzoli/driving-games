from typing import Tuple
import do_mpc
from dg_commons.sim.models.vehicle import VehicleState

from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from homotopies.utils import *

vehicle_params = VehicleParameters.default_car()


@dataclass
class MpccKinBaseParams:
    n_horizon: int = 30
    """ Horizon Length """
    t_step: float = 0.1
    """ Sample Time """
    cost = QuadraticCost
    """ Cost function """
    cost_params: Union[List[CostParameters], CostParameters] = QuadraticParams(
        q=SemiDef(eig=[1, 1, 1]),
        r=SemiDef(matrix=np.eye(2))
    )
    s_reward = -1
    """ Cost function parameters """
    delta_input_weight: float = 0
    """ Weighting factor in cost function for varying input """
    vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()

    v_delta_bounds: Tuple[float, float] = (-vehicle_params.ddelta_max, vehicle_params.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Tuple[float, float] = (
    -vehicle_params.default_car().delta_max, vehicle_params.default_car().delta_max)
    """ Steering Bounds """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits  # longitudinal speed in vehicle frame

    suppress_ipopt = {'ipopt.print_level': 0,
                      'ipopt.file_print_level': 5,
                      'ipopt.sb': 'yes',
                      'print_time': 0,
                      #'ipopt.linear_solver': 'MA27',
                      'ipopt.max_iter': 100,
                      'ipopt.output_file': 'ipopt_log.txt',
                      #'ipopt.mu_strategy': 'adaptive'
                      }


class MpccKinBase:
    def __init__(self, params, model_type: str):
        self.params = params
        self.model = do_mpc.model.Model(model_type)

        self.state_x = self.model.set_variable(var_type="_x", var_name="state_x", shape=(1, 1))
        self.state_y = self.model.set_variable(var_type="_x", var_name="state_y", shape=(1, 1))
        self.theta = self.model.set_variable(var_type="_x", var_name="theta", shape=(1, 1))
        self.v = self.model.set_variable(var_type="_x", var_name="v", shape=(1, 1))
        self.delta = self.model.set_variable(var_type="_x", var_name="delta", shape=(1, 1))
        self.s_des = self.model.set_variable(var_type="_x", var_name="s_des", shape=(1, 1))

        self.a = self.model.set_variable(var_type="_u", var_name="a")
        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')
        self.vs_des = self.model.set_variable(var_type='_u', var_name='vs_des')

        self.obstacle_obs: VehicleState = None
        self.obstacle_obs_flag = False
        self.obstacle_state = self.model.set_variable(var_type='_tvp', var_name='obstacle_state', shape=(5, 1))

        self.homotopy = self.model.set_variable('_p',
                                                'homotopy')  # 0 for overtaking from left, 1 for overtaking from right

        self.cost = self.params.cost(self.params.cost_params)

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'use_terminal_bounds': True,
            'store_full_solution': True,
            #'nl_cons_check_colloc_points': False,
            'nlpsol_opts': self.params.suppress_ipopt,
        }
