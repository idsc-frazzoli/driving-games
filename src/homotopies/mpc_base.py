from dataclasses import dataclass
from typing import Tuple
import do_mpc
import numpy as np
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
    path_approx_technique: Union[List[CurveApproximationTechnique], CurveApproximationTechnique] = LinearCurve
    """ Path approximation technique """
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


class MpcKinBase(LatAndLonController, ABC):
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

        self.target_speed = self.model.set_variable(var_type='_tvp', var_name='target_speed', shape=(1, 1))
        self.path_approx: CurveApproximationTechnique = self.params.path_approx_technique()
        self.path_params = self.model.set_variable(var_type='_tvp', var_name='path_params',
                                                   shape=(self.path_approx.n_params, 1))
        self.path_parameters = self.path_approx.n_params * [0]

        self.homotopy_classes = self.model.set_variable('_p', 'homotopy')#0 for overtaking from left, 1 for overtaking from right

        self.cost = self.params.cost(self.params.cost_params)

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }



    def _update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref

    def _get_acceleration(self, at: float) -> float:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        try:
            return self.u[2][0]
        except IndexError:
            return self.u[1][0]

    def _get_steering(self, at: float):
        return self.u[0][0]

    def _update_obs(self, new_obs: Optional[X] = None):
        self.current_position = np.array([new_obs.x, new_obs.y])
        self.current_speed = new_obs.vx
        control_sol_params = self.control_path.ControlSolParams(self.current_speed, self.params.t_step)
        self.current_beta, _ = self.control_path.find_along_lane_closest_point(self.current_position,
                                                                               control_sol=control_sol_params)
        """ Update current state of the vehicle """
        pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
        self.path_approx.update_from_data(pos1, angle1, pos2, angle2, pos3, angle3)
        params = self.path_approx.parameters
        """ Generate current path approximation """
        self.path_parameters = params[:self.path_approx.n_params]

        x0_temp = [self.current_position[0], self.current_position[1], new_obs.theta, self.current_speed, new_obs.delta]
        x0_temp = x0_temp if self.params.analytical else x0_temp + [pos1[0]]
        x0 = np.array(x0_temp).reshape(-1, 1)
        """ Define initial condition """
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.u = self.mpc.make_step(x0)
        """ Compute input """


