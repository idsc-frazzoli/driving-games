import math
from dataclasses import dataclass
from typing import Optional, Mapping, Callable
from abc import ABC, abstractmethod
from geometry import translation_angle_from_SE2
from dg_commons.planning.lanes import DgLanelet
from games import X
import do_mpc
from casadi import *
import matplotlib.pyplot as plt
from sim.models.vehicle_structures import VehicleGeometry


__all__ = ["MPCBase", "MPCBAseParam"]


@dataclass
class MPCBAseParam:
    n_horizon: int = 15
    """ Horizon Length """
    t_step: float = 0.1
    """ Sample Time """
    state_mult: float = 1
    """ Weighting factor in cost function for having state error """
    input_mult: float = 1
    """ Weighting factor in cost function for applying input """
    delta_input_mult: float = 1e-2
    """ Weighting factor in cost function for varying input """


class MPCBase(ABC):

    def _get_linear_func(self, beta):
        beta1, beta2 = beta, beta + 1
        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)

        x1, y1 = pos1[0], pos1[1]
        x2, y2 = pos2[0], pos2[1]
        plt.xlim(-10, 10), plt.ylim(-10, 10)
        plt.plot(x1, y1, x2, y2, marker='o')
        plt.show()

        vertical_line = False
        if pos1[0] == pos2[0] and pos1[1] != pos2[2]:
            vertical_line = True
            res = [pos1[0], 1 if pos1[1] < pos2[1] else -1]
        else:
            m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
            b = pos1[1] - m * pos1[0]
            res = [m, b]

        def func(x):
            return res[0]*x + res[1]

        return func

    def _get_cubic_func(self, beta):
        beta1, beta2 = beta, beta+1
        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)

        A = np.array([[pos1[0]**3, pos1[0]**2, pos1[0], 1], [pos2[0]**3, pos2[0]**2, pos2[0], 1],
                     [3*pos1[0]**2, 2*pos1[0], 1, 0], [3*pos2[0]**2, 2*pos2[0], 1, 0]])
        b = np.array([[pos1[1]], [pos2[1]], [math.tan(angle1)], [math.tan(angle2)]])

        res = np.linalg.solve(A, b)

        def func(x):
            return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x + res[3]

        return func

    def _get_quadratic_func(self, beta):
        beta1, beta2 = beta, beta+1
        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)

        A = np.array([[pos1[0]**2, pos1[0], 1], [pos2[0]**2, pos2[0], 1], [2*pos1[0], 1, 0]])
        b = np.array([[pos1[1]], [pos2[1]], [math.tan(angle1)]])

        res = np.linalg.solve(A, b)

        def func(x):
            return res[0] * x ** 2 + res[1] * x + res[2]

        return func

    techniques: Mapping[str, Callable[[float], Callable[[float], float]]] = \
        {'linear': _get_linear_func, 'cubic': _get_cubic_func, 'quadratic': _get_quadratic_func}

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_state(self, obs: X, speed_ref: Optional[float] = None):
        position = np.array([obs.x, obs.y])
        current_beta, _ = self.path.find_along_lane_closest_point(position)
        s0, _ = translation_angle_from_SE2(self.path.center_point(current_beta))

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)

        target_x, target_y, target_angle = self.compute_targets(current_beta)
        lterm = self.lterm(target_x, target_y, speed_ref)
        mterm = self.mterm(target_x, target_y, speed_ref)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_mult
        )

        self.set_bounds()
        self.set_scaling()

        self.mpc.setup()

        x0 = np.array([obs.x, obs.y, obs.theta, obs.vx, obs.delta, s0[0]]).reshape(-1, 1) if self.path_var else \
            np.array([obs.x, obs.y, obs.theta, obs.vx, obs.delta]).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.u = self.mpc.make_step(x0)

    @abstractmethod
    def __init__(self, params, model_type: str):
        self.model = None
        self.path: Optional[DgLanelet] = None
        self.params = params
        self.vehicle_geometry = VehicleGeometry.default_car()
        self.traj = None
        self.u = None
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
        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')

    @abstractmethod
    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def compute_targets(self, current_beta):
        pass

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def set_scaling(self):
        pass
