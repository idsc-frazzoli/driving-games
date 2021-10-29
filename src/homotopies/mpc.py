from dataclasses import dataclass

from casadi import *

__all__ = ["MpcFullKinCont", "NMPCFullKinContParam"]

from homotopies.mpc_base import MpcKinBase


@dataclass
class NMPCFullKinContParam:
    pass


class MpcFullKinCont(MpcKinBase):
    """ Nonlinear MPC contouring control of vehicle. Kinematic model without prior discretization """

    def __init__(self, params: NMPCFullKinContParam):
        model_type = "continuous"  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        dtheta = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        vy = dtheta * self.params.vehicle_geometry.lr
        self.model.set_rhs("state_x", self.v * cos(self.theta) - vy * sin(self.theta))
        self.model.set_rhs("state_y", self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs("theta", dtheta)
        self.model.set_rhs("v", self.a)
        self.model.set_rhs("delta", self.v_delta)
        self.model.set_rhs("s", self.v_s)

        self.model.setup()
        self.set_up_mpc()

    def compute_targets(self):
        if self.params.analytical:
            self.path_approx.update_from_parameters(self.path_params)
            return *self.path_approx.closest_point_on_path([self.state_x, self.state_y]), None
        else:
            self.path_approx.update_from_parameters(self.path_params)
            return self.s, self.path_approx.function(self.s), None

    def set_scaling(self):
        self.mpc.scaling["_x", "x_x"] = 1
        self.mpc.scaling["_x", "x_y"] = 1
        self.mpc.scaling["_x", "x_theta"] = 1
        self.mpc.scaling["_x", "x_v"] = 1
        self.mpc.scaling["_x", "x_delta"] = 1
        self.mpc.scaling["_u", "u_ddelta"] = 1
        self.mpc.scaling["_u", "u_acc"] = 1
