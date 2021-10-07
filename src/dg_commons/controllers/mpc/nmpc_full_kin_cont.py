from dg_commons.controllers.mpc.full_mpc_base import FullMPCKinBasePathVariable, FullMPCKinBaseParam, \
    FullMPCKinBaseAnalytical
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
        dtheta = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        if self.params.rear_axle:
            self.model.set_rhs('state_x', cos(self.theta) * self.v)
            self.model.set_rhs('state_y', sin(self.theta) * self.v)
        else:
            vy = dtheta * self.params.vehicle_geometry.lr
            self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
            self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

        self.model.setup()
        self.set_up_mpc()

    def compute_targets(self):
        res, current_trajectory = self.techniques[self.approx_type][1](self, self.path_params)
        return self.s, current_trajectory(self.s), None

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
        self.mpc.scaling['_u', 'a'] = 1

    def get_targets(self):
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
        dtheta = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        if self.params.rear_axle:
            self.model.set_rhs('state_x', cos(self.theta) * self.v)
            self.model.set_rhs('state_y', sin(self.theta) * self.v)
        else:
            vy = dtheta * self.params.vehicle_geometry.lr
            self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
            self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        self.model.setup()
        self.set_up_mpc()

    def compute_targets(self):
        current_trajectory = self.techniques[self.approx_type][1](self, self.path_params)
        return current_trajectory(self.state_x, self.state_y)

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1

    def get_targets(self):
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0], self.u[1][0]
