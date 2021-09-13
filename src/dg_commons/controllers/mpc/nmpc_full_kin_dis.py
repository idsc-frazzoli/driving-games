from dataclasses import dataclass
from casadi import *
from dg_commons.controllers.mpc.mpc_base import MPCBase, MPCBAseParam
from dg_commons.controllers.mpc.discretization_techniques import kin_euler, discretizations
from typing import Tuple


__all__ = ["NMPCFullKinDis", "NMPCFullKinDisParam"]


@dataclass
class NMPCFullKinDisParam(MPCBAseParam):
    technique: str = 'linear'
    """ Path approximation technique """
    speed_mult: float = 1
    """ Weighting factor in cost function for velocity error """
    acc_mult: float = 1
    """ Weighting factor in cost function for acceleration """
    dis_technique: str = 'Kinematic Euler'
    """ Discretization technique """
    dis_t: float = 0.01
    """ Discretization Time Step """
    acc_bounds: Tuple[float, float] = (-8, 5)
    """ Accelertion bounds """


class NMPCFullKinDis(MPCBase):
    """ Nonlinear MPC for full control of vehicle. Kinematic model with prior discretization """

    def __init__(self, params: NMPCFullKinDisParam = NMPCFullKinDisParam()):
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        assert self.params.technique in self.techniques.keys()

        self.s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))
        self.v_s = self.model.set_variable(var_type='_u', var_name='v_s')
        self.a = self.model.set_variable(var_type='_u', var_name='a')

        f = [self.state_x, self.state_y, self.theta, self.v, self.delta, self.s]
        for _ in range(int(self.params.t_step/self.params.dis_t)):
            f = discretizations[self.params.dis_technique](f[0], f[1], f[2], f[3], f[4], f[5], self.v_delta,
                                                           self.v_s, self.a, self.vehicle_geometry, self.params.dis_t)

        self.path_var = True
        # Set right right hand side of difference equation for x, y, theta, v, delta and s
        self.model.set_rhs('state_x', f[0])
        self.model.set_rhs('state_y', f[1])
        self.model.set_rhs('theta', f[2])
        self.model.set_rhs('v', f[3])
        self.model.set_rhs('delta', f[4])
        self.model.set_rhs('s', f[5])

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.state_mult * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.speed_mult * (self.v - speed_ref)**2 + \
               self.params.input_mult * self.v_delta ** 2 + \
               self.params.acc_mult * self.a ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.state_mult * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.speed_mult * (self.v - speed_ref)**2

    def compute_targets(self, current_beta):
        self.traj = self.techniques[self.params.technique](self, current_beta)
        return self.s, self.traj(self.s), None

    def set_bounds(self):
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        # self.mpc.scaling['_x', 's'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
        self.mpc.scaling['_u', 'a'] = 1

    def get_targets(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0], self.u[2][0]
