import math
from dataclasses import dataclass
from typing import Optional

import casadi
import numpy as np

from sim.models.vehicle_structures import VehicleGeometry
from geometry import SE2value, SE2_from_translation_angle, translation_angle_from_SE2
from dg_commons.planning.lanes import DgLanelet
from games import X, U
import do_mpc
from do_mpc.data import save_results
from casadi import *
import matplotlib.pyplot as plt
import matplotlib as mpl

__all__ = ["MPCKinContPathVar", "MPCKinContPathVarParam"]


@dataclass
class MPCKinContPathVarParam:
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
    technique: str = 'linear'
    """ Path approximation technique """


class MPCKinContPathVar:

    def __init__(self, params: MPCKinContPathVarParam = MPCKinContPathVarParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.techniques = {'linear': self.get_linear_func, 'cubic': self.get_cubic_func,
                           'quadratic': self.get_quadratic_func}
        self.params: MPCKinContPathVarParam = params
        assert self.params.technique in self.techniques.keys()

        self.path: Optional[DgLanelet] = None
        self.vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()
        self.u: float = 0
        self.traj = None

        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        self.state_x = self.model.set_variable(var_type='_x', var_name='state_x', shape=(1, 1))
        self.state_y = self.model.set_variable(var_type='_x', var_name='state_y', shape=(1, 1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        self.v = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        self.delta = self.model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))
        self.s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))

        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')
        self.v_s = self.model.set_variable(var_type='_u', var_name='v_s')

        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)

        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

        self.model.setup()
        self.mpc = None
        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_state(self, obs: X):
        position = np.array([obs.x, obs.y])
        current_beta, _ = self.path.find_along_lane_closest_point(position)
        s0, _ = translation_angle_from_SE2(self.path.center_point(current_beta))
        self.traj = self.techniques[self.params.technique](current_beta)

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)

        y = self.traj(self.s)
        lterm = self.params.state_mult * ((self.s - self.state_x) ** 2 + (y - self.state_y) ** 2) + \
                self.params.input_mult * self.v_delta ** 2

        mterm = (self.s - self.state_x) ** 2 + (y - self.state_y) ** 2

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(
            v_delta=self.params.delta_input_mult
        )

        self.mpc.bounds['lower', '_u', 'v_delta'] = -1
        self.mpc.bounds['upper', '_u', 'v_delta'] = 1
        self.mpc.bounds['lower', '_x', 'delta'] = -0.52
        self.mpc.bounds['upper', '_x', 'delta'] = 0.52

        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        #self.mpc.scaling['_x', 's'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1

        self.mpc.setup()

        x0 = np.array([obs.x, obs.y, obs.theta, obs.vx, obs.delta, s0[0]]).reshape(-1, 1)
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.u = self.mpc.make_step(x0)

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0]

    def get_linear_func(self, beta):
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

    def get_cubic_func(self, beta):
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

    def get_quadratic_func(self, beta):
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
