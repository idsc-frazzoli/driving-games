from dataclasses import dataclass
from casadi import *
from dg_commons.controllers.mpc.mpc_base_classes import FullMPCBasePathVariable, FullMPCBAseParam
from typing import Tuple


__all__ = ["NMPCFullKinCont", "NMPCFullKinContParam"]


@dataclass
class NMPCFullKinContParam(FullMPCBAseParam):
    technique: str = 'linear'
    """ Path approximation technique """


class NMPCFullKinCont(FullMPCBasePathVariable):
    """ Nonlinear MPC for full control of vehicle. Kinematic model without prior discretization """

    def __init__(self, params: NMPCFullKinContParam = NMPCFullKinContParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

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
