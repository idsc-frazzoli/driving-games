from dg_commons.controllers.mpc.lateral_mpc_base import LatMPCKinBasePathVariable, LatMPCKinBaseParam, \
    LatMPCKinBaseAnalytical
from dg_commons.controllers.mpc.mpc_utils.cost_functions import *


__all__ = ["NMPCLatKinContPV", "NMPCLatKinContPVParam", "NMPCLatKinContAN", "NMPCLatKinContANParam"]


@dataclass
class NMPCLatKinContPVParam(LatMPCKinBaseParam):
    path_approx_technique: str = 'linear'
    """ Path approximation technique """


class NMPCLatKinContPV(LatMPCKinBasePathVariable):
    """ Nonlinear MPC for lateral control of vehicle. Kinematic model without prior discretization """

    def __init__(self, params: NMPCLatKinContPVParam = NMPCLatKinContPVParam()):
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
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

        self.model.setup()

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

    def compute_targets(self, current_beta):
        res, self.traj, vertical_line = self.techniques[self.params.path_approx_technique](self, current_beta)
        target_x = res[0] if vertical_line else self.s
        target_y = self.state_y if vertical_line else self.traj(self.s)
        return target_x, target_y, None

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        #self.mpc.scaling['_x', 's'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0]


@dataclass
class NMPCLatKinContANParam(LatMPCKinBaseParam):
    path_approx_technique: str = 'linear'
    """ Path Approximation Technique """


class NMPCLatKinContAN(LatMPCKinBaseAnalytical):

    def __init__(self, params: NMPCLatKinContANParam = NMPCLatKinContANParam()):
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
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)

        self.model.setup()

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

    def compute_targets(self, current_beta):
        self.traj = self.techniques[self.params.path_approx_technique](self, current_beta)
        return self.traj(self.state_x, self.state_y)

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        # todo fixme this controller is not precise, as we use the cog rather than the base link
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        return self.u[0][0]

