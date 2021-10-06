from typing import Optional, Mapping, Callable, Tuple, List
from abc import abstractmethod, abstractproperty
from geometry import translation_angle_from_SE2, SE2_from_translation_angle
from dg_commons.maps.lanes import DgLanelet
from dg_commons import X
import do_mpc
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters
from dg_commons.controllers.mpc.mpc_base import MPCKinBAseParam, MPCKinBase
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.controllers.path_approximation_techniques import *
from duckietown_world.utils import SE2_apply_R2
import time


VEHICLE_PARAMS = VehicleParameters.default_car()


@dataclass
class LatMPCKinBaseParam(MPCKinBAseParam):
    cost: str = "quadratic"
    """ Cost function """
    cost_params: CostParameters = QuadraticParams(
        q=SemiDef(matrix=np.eye(2)),
        r=SemiDef(matrix=np.eye(1))
    )
    """ Cost function parameters """
    delta_input_weight: float = 1e-2
    """ Weighting factor in cost function for varying input """
    v_delta_bounds: Tuple[float, float] = (-VEHICLE_PARAMS.ddelta_max, VEHICLE_PARAMS.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Tuple[float, float] = (-VEHICLE_PARAMS.delta_max, VEHICLE_PARAMS.delta_max)
    """ Steering Bounds """
    path_approx_technique: str = 'linear'
    """ Path approximation technique """


class LatMPCKinBase(MPCKinBase):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.path: Optional[DgLanelet] = None
        self.vehicle_geometry = VehicleGeometry.default_car()
        self.traj = None
        self.u = None
        self.path_var = False
        self.current_speed = None
        self.current_beta = None

        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')
        self.n_params = int(self.techniques[self.params.path_approx_technique][2])
        self.par = self.model.set_variable(var_type='_x', var_name='beta', shape=(self.n_params, 1))
        self.nominal_speed = self.model.set_variable(var_type='_x', var_name='speed_ref', shape=(1, 1))

        self.current_position = None
        self.target_position = None
        self.current_f = None

        self.prediction_x = None
        self.prediction_y = None
        self.vertical = False

    def set_up_mpc(self):
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)
        target_x, target_y, target_angle = self.compute_targets()

        lterm = self.lterm(target_x, target_y, self.nominal_speed)
        mterm = self.mterm(target_x, target_y, self.nominal_speed)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )

        self.set_bounds()
        self.set_scaling()

        self.mpc.setup()

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def rear_axle_position(self, obs: X):
        pose = SE2_from_translation_angle([obs.x, obs.y], obs.theta)
        return SE2_apply_R2(pose, np.array([-self.vehicle_geometry.lr, 0]))

    def cog_position(self, obs: X):
        return np.array([obs.x, obs.y])

    def update_state(self, obs: Optional[X] = None, speed_ref: float = 0):
        self.current_position = self.rear_axle_position(obs) if self.params.rear_axle else self.cog_position(obs)
        self.speed_ref = speed_ref
        self.current_speed = obs.vx
        control_sol_params = self.path.ControlSolParams(self.current_speed, self.params.t_step)
        self.current_beta, _ = self.path.find_along_lane_closest_point(self.current_position,
                                                                       control_sol=control_sol_params)

        s0, _ = translation_angle_from_SE2(self.path.center_point(self.current_beta))
        pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
        params = list(self.techniques[self.params.path_approx_technique][0](pos1, angle1, pos2, angle2, pos3, angle3))
        self.vertical = params[-1]

        x_add = params[:self.n_params] + [speed_ref]
        x0 = np.array([self.current_position[0], self.current_position[1], obs.theta,
                       self.current_speed, obs.delta] + x_add + [s0[0]]).reshape(-1, 1) if self.path_var else \
             np.array([self.current_position[0], self.current_position[1], obs.theta,
                       self.current_speed, obs.delta] + x_add).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

        self.u = self.mpc.make_step(x0)

        self.prediction_x = self.mpc.data.prediction(('_x', 'state_x', 0))[0]
        self.prediction_y = self.mpc.data.prediction(('_x', 'state_y', 0))[0]

    def next_pos(self, current_beta):
        along_lane = self.path.along_lane_from_beta(current_beta)
        delta_step = self.delta_step()
        along_lane1 = along_lane + delta_step/2
        along_lane2 = along_lane1 + delta_step/2

        beta1, beta2, beta3 = current_beta, self.path.beta_from_along_lane(along_lane1), \
                              self.path.beta_from_along_lane(along_lane2)

        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)
        q3 = self.path.center_point(beta3)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)
        pos3, angle3 = translation_angle_from_SE2(q3)

        self.target_position = pos3
        return pos1, angle1, pos2, angle2, pos3, angle3

    def delta_step(self):
        return self.current_speed*self.params.t_step*self.params.n_horizon

    @abstractmethod
    def compute_targets(self):
        pass

    def set_bounds(self):
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]

    @property
    @abstractmethod
    def techniques(self) -> Mapping[str, Tuple[Callable, Callable, int]]:
        pass


class LatMPCKinBaseAnalytical(LatMPCKinBase):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        assert self.params.path_approx_technique in self.techniques.keys()

    def _get_linear(self, params):
        res, f, _, closest_point = linear(params[0, 0], params[1, 0], params[2, 0])

        self.current_f = f

        def func(x, y):
            pos = closest_point([x, y])
            return pos[0], pos[1], res[2]

        return func

    def _get_quadratic(self, params):
        a, b, c = params[0, 0], params[1, 0], params[2, 0]
        res, f, vertical_line, closest_point = quadratic(a, b, c)
        self.current_f = f

        if vertical_line:
            def func(x, y):
                pos = closest_point([x, y])
                return pos[0], pos[1], res[2]
        else:
            '''if abs(2*a*pos2[0]) / abs(2*a*pos2[0] + b) < 5*10e-2:
                return self._get_linear(beta)'''

            def func(x, y):
                pos = closest_point([x, y])
                return pos[0], pos[1], None

        return func

    techniques = {'linear': [linear_param, _get_linear, 3], 'quadratic': [quadratic_param, _get_quadratic, 3]}


class LatMPCKinBasePathVariable(LatMPCKinBase):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        assert self.params.path_approx_technique in self.techniques.keys()

        self.path_var = True
        self.s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))
        self.v_s = self.model.set_variable(var_type='_u', var_name='v_s')

    def _get_linear_func(self, params):
        res, func, vertical, _ = linear(params[0, 0], params[1, 0], params[2, 0])
        self.current_f = func

        return res, func, vertical

    def _get_cubic_func(self, params):
        res, func, vertical, _ = cubic(params[0, 0], params[1, 0], params[2, 0], params[3, 0])
        self.current_f = func

        return res, func, vertical

    def _get_quadratic_func(self, params):
        res, func, vertical, _ = quadratic(params[0, 0], params[1, 0], params[2, 0])
        self.current_f = func

        return res, func, vertical

    techniques = {'linear': [linear_param, _get_linear_func, 3], 'cubic': [cubic_param, _get_cubic_func, 4],
                  'quadratic': [quadratic_param, _get_quadratic_func, 3]}
