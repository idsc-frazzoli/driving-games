from dataclasses import dataclass

from casadi import *

__all__ = ["MpcFullKinCont", "NMPCFullKinContParam"]

from homotopies.mpc_base import MpcKinBase, MpcKinBaseParams


@dataclass
class NMPCFullKinContParam:
    pass


class MpcFullKinCont(MpcKinBase):
    """ Nonlinear MPC contouring control of vehicle. Kinematic model without prior discretization """
    USE_STEERING_VELOCITY: bool = True

    def __init__(self, params: MpcKinBaseParams = MpcKinBaseParams()):
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

        self.model.setup()
        self.set_up_mpc()
