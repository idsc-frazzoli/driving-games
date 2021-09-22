from dataclasses import dataclass
from casadi import *
from dg_commons.controllers.mpc.lateral_mpc_base import LatMPCKinBasePathVariable, LatMPCKinBaseParam, \
    LatMPCKinBaseAnalytical

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
        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.position_err_weight * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.steering_vel_weight * self.v_delta ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return (target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2

    def compute_targets(self, current_beta):
        self.traj = self.techniques[self.params.path_approx_technique](self, current_beta)
        return self.s, self.traj(self.s), None

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
        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.position_err_weight * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
                self.params.steering_vel_weight * self.v_delta ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return (target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2

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

