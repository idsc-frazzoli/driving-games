from typing import Tuple
from geometry import translation_angle_from_SE2, SE2_from_translation_angle
from dg_commons.maps.lanes import DgLanelet
from dg_commons import X
from dg_commons.controllers.mpc.mpc_base import MPCKinBAseParam, MPCKinBase
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.controllers.path_approximation_techniques import *
from duckietown_world.utils import SE2_apply_R2
from sim.models.vehicle_utils import VehicleParameters
from dg_commons.controllers.path_approximation_techniques import PathApproximationTechniques, LinearPath

vehicle_params = VehicleParameters.default_car()


@dataclass
class LatMPCKinBaseParam(MPCKinBAseParam):
    cost: str = "quadratic"
    """ Cost function """
    cost_params: CostParameters = QuadraticParams(
        q=SemiDef(matrix=np.eye(2)),
        r=SemiDef(matrix=np.eye(1))
    )
    """ Cost function parameters """
    v_delta_bounds: Tuple[float, float] = (-vehicle_params.ddelta_max,
                                           vehicle_params.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Tuple[float, float] = (-vehicle_params.default_car().delta_max,
                                         vehicle_params.default_car().delta_max)
    """ Steering Bounds """
    path_approx_technique: PathApproximationTechniques = LinearPath()
    """ Path approximation technique """
    analytical: bool = False


class LatMPCKinBase(MPCKinBase):
    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.path: Optional[DgLanelet] = None
        """ Referece DgLanelet path """
        self.u = None
        """ Current input to the system """

        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')
        self.path_approx: PathApproximationTechniques = self.params.path_approx_technique()
        self.path_params = self.model.set_variable(var_type='_tvp', var_name='path_params',
                                                   shape=(self.path_approx.n_params, 1))
        self.path_parameters = self.path_approx.n_params*[0]

        self.current_position, self.current_speed, self.current_beta, self.current_f = None, None, None, None
        """ Current position and speed of the vehicle, current position on DgLanelet and current approx trajectory """

        self.prediction_x, self.prediction_y, self.target_position = None, None, None

        self.approx_type = self.params.path_approx_technique
        if not self.params.analytical:
            self.s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))
            self.v_s = self.model.set_variable(var_type='_u', var_name='v_s')

    def func(self, t_now):
        temp = [self.speed_ref] + self.path_parameters
        self.tvp_temp['_tvp', :] = np.array(temp)
        return self.tvp_temp

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def rear_axle_position(self, obs: X):
        pose = SE2_from_translation_angle([obs.x, obs.y], obs.theta)
        return SE2_apply_R2(pose, np.array([-self.params.vehicle_geometry.lr, 0]))

    @staticmethod
    def cog_position(obs: X):
        return np.array([obs.x, obs.y])

    def update_state(self, obs: Optional[X] = None, speed_ref: float = 0):
        self.current_position = self.rear_axle_position(obs) if self.params.rear_axle else self.cog_position(obs)
        self.current_speed = obs.vx
        control_sol_params = self.path.ControlSolParams(self.current_speed, self.params.t_step)
        self.current_beta, _ = self.path.find_along_lane_closest_point(self.current_position,
                                                                       control_sol=control_sol_params)
        """ Update current state of the vehicle """
        pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
        self.path_approx.update_from_data(pos1, angle1, pos2, angle2, pos3, angle3)
        params = self.path_approx.parameters
        """ Generate current path approximation """
        self.speed_ref = speed_ref
        self.path_parameters = params[:self.path_approx.n_params]

        x0_temp = [self.current_position[0], self.current_position[1], obs.theta, self.current_speed, obs.delta]
        x0_temp = x0_temp if self.params.analytical else x0_temp + [pos1[0]]
        x0 = np.array(x0_temp).reshape(-1, 1)
        """ Define initial condition """
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.u = self.mpc.make_step(x0)
        """ Compute input """

        self.store_extra()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y]
        inp = [self.v_delta]

        lterm, _ = costs[self.params.cost](error, inp, self.params.cost_params)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y]
        inp = [self.v_delta]

        _, mterm = costs[self.params.cost](error, inp, self.params.cost_params)
        return mterm

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

    def set_bounds(self):
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]

    def store_extra(self):
        self.prediction_x = self.mpc.data.prediction(('_x', 'state_x', 0))[0]
        self.prediction_y = self.mpc.data.prediction(('_x', 'state_y', 0))[0]
