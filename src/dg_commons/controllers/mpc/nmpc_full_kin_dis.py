from dataclasses import dataclass
from casadi import *
from dg_commons.controllers.mpc.full_mpc_base import FullMPCKinBasePathVariable, FullMPCKinBaseParam
from dg_commons.controllers.mpc.discretization_techniques import kin_euler, discretizations
from typing import Tuple


__all__ = ["NMPCFullKinDisPV", "NMPCFullKinDisPVParam"]


@dataclass
class NMPCFullKinDisPVParam(FullMPCKinBaseParam):
    path_approx_technique: str = 'linear'
    """ Path approximation technique """
    dis_technique: str = 'Kinematic Euler'
    """ Discretization technique """
    dis_t: float = 0.01
    """ Discretization Time Step """


class NMPCFullKinDisPV(FullMPCKinBasePathVariable):
    """ Nonlinear MPC for full control of vehicle. Kinematic model with prior discretization """

    def __init__(self, params: NMPCFullKinDisPVParam = NMPCFullKinDisPVParam()):
        model_type = 'discrete'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        assert self.params.dis_technique in discretizations.keys()
        assert self.params.t_step % self.params.dis_t < 10e-10

        f = [self.state_x, self.state_y, self.theta, self.v, self.delta, self.s]
        for _ in range(int(self.params.t_step/self.params.dis_t)):
            f = discretizations[self.params.dis_technique](f[0], f[1], f[2], f[3], f[4], f[5], self.v_delta,
                                                           self.v_s, self.a, self.vehicle_geometry, self.params.dis_t)

        # Set right right hand side of difference equation for x, y, theta, v, delta and s
        self.model.set_rhs('state_x', f[0])
        self.model.set_rhs('state_y', f[1])
        self.model.set_rhs('theta', f[2])
        self.model.set_rhs('v', f[3])
        self.model.set_rhs('delta', f[4])
        self.model.set_rhs('s', f[5])

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.position_err_weight * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.velocity_err_weight * (self.v - speed_ref)**2 + \
               self.params.steering_vel_weight * self.v_delta ** 2 + \
               self.params.acceleration_weight * self.a ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.position_err_weight * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.velocity_err_weight * (self.v - speed_ref)**2

    def compute_targets(self, current_beta):
        self.traj = self.techniques[self.params.path_approx_technique](self, current_beta)
        return self.s, self.traj(self.s), None

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
