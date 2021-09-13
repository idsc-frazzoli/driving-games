from dataclasses import dataclass
from casadi import *
from dg_commons.controllers.mpc.mpc_base import MPCBase, MPCBAseParam

__all__ = ["MPCKinContPathVar", "MPCKinContPathVarParam"]


@dataclass
class MPCKinContPathVarParam(MPCBAseParam):
    technique: str = 'linear'
    """ Path approximation technique """


class MPCKinContPathVar(MPCBase):

    def __init__(self, params: MPCKinContPathVarParam = MPCKinContPathVarParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        assert self.params.technique in self.techniques.keys()

        self.s = self.model.set_variable(var_type='_x', var_name='s', shape=(1, 1))
        self.v_s = self.model.set_variable(var_type='_u', var_name='v_s')

        self.path_var = True
        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)
        self.model.set_rhs('s', self.v_s)

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.state_mult * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
               self.params.input_mult * self.v_delta ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return (target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2

    def compute_targets(self, current_beta):
        self.traj = self.techniques[self.params.technique](self, current_beta)
        return self.s, self.traj(self.s), None

    def set_bounds(self):
        self.mpc.bounds['lower', '_u', 'v_delta'] = -1
        self.mpc.bounds['upper', '_u', 'v_delta'] = 1
        self.mpc.bounds['lower', '_x', 'delta'] = -0.52
        self.mpc.bounds['upper', '_x', 'delta'] = 0.52

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
