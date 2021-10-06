import casadi

from dg_commons.controllers.mpc.full_mpc_base import FullMPCKinBasePathVariable, FullMPCKinBaseParam, \
    FullMPCKinBaseAnalytical
from typing import Tuple
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *


__all__ = ["NMPCFullKinContPV", "NMPCFullKinContPVParam", "NMPCFullKinContAN", "NMPCFullKinContANParam"]


@dataclass
class NMPCFullKinContPVParam(FullMPCKinBaseParam):
    pass


class NMPCFullKinContPV(FullMPCKinBasePathVariable):
    """ Nonlinear MPC for full control of vehicle. Kinematic model without prior discretization """

    def __init__(self, params: NMPCFullKinContPVParam = NMPCFullKinContPVParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        dtheta = self.v * tan(self.delta) / self.vehicle_geometry.length
        if self.params.rear_axle:
            self.model.set_rhs('state_x', cos(self.theta) * self.v)
            self.model.set_rhs('state_y', sin(self.theta) * self.v)
        else:
            vy = dtheta * self.vehicle_geometry.lr
            self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
            self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)
        self.model.set_rhs('beta', SX.zeros(self.n_params, 1))
        self.model.set_rhs('speed_ref', casadi.SX(0))
        self.model.setup()
        self.set_up_mpc()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]

        lterm, _ = costs[self.params.cost](error, inp, self.params.cost_params)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]

        _, mterm = costs[self.params.cost](error, inp, self.params.cost_params)
        return mterm

    def compute_targets(self):
        res, self.traj, vertical_line = self.techniques[self.params.path_approx_technique][1](self, self.par)
        '''target_x = res[0] if vertical_line else self.s
        target_y = self.state_y if vertical_line else self.traj(self.s)
        return target_x, target_y, None'''
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

    def get_targets(self) -> Tuple[float, float]:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0], self.u[2][0]


@dataclass
class NMPCFullKinContANParam(FullMPCKinBaseParam):
    pass


class NMPCFullKinContAN(FullMPCKinBaseAnalytical):

    def __init__(self, params: NMPCFullKinContANParam = NMPCFullKinContANParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        self.path_var = False
        assert self.params.path_approx_technique in self.techniques.keys()

        # Set right right hand side of differential equation for x, y, theta, v, and delta
        dtheta = self.v * tan(self.delta) / self.vehicle_geometry.length
        if self.params.rear_axle:
            self.model.set_rhs('state_x', cos(self.theta) * self.v)
            self.model.set_rhs('state_y', sin(self.theta) * self.v)
        else:
            vy = dtheta * self.vehicle_geometry.lr
            self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
            self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('beta', SX.zeros(self.n_params, 1))
        self.model.set_rhs('speed_ref', casadi.SX(0))
        self.model.setup()
        self.set_up_mpc()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]

        lterm, _ = costs[self.params.cost](error, inp, self.params.cost_params)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]

        _, mterm = costs[self.params.cost](error, inp, self.params.cost_params)
        return mterm

    def compute_targets(self):
        self.traj = self.techniques[self.params.path_approx_technique][1](self, self.par)
        return self.traj(self.state_x, self.state_y)

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1

    def get_targets(self) -> Tuple[float, float]:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0], self.u[1][0]
