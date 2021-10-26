from dataclasses import dataclass
from typing import Tuple
import do_mpc
import numpy as np


@dataclass
class FullMPCKinBaseParam:
    cost: CostFunctions = QuadraticCost
    """ Cost function """
    cost_params: CostParameters = QuadraticParams(q=SemiDef(matrix=np.eye(3)), r=SemiDef(matrix=np.eye(2)))
    """ Cost function parameters """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Acceleration bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits


class MpcKinBase(LatMPCKinBase, LatAndLonController):
    def __init__(self, params, model_type: str):
        self.model = do_mpc.model.Model(model_type)
        self.state_x = self.model.set_variable(var_type="_x", var_name="x_x", shape=(1, 1))
        self.state_y = self.model.set_variable(var_type="_x", var_name="x_y", shape=(1, 1))
        self.theta = self.model.set_variable(var_type="_x", var_name="x_theta", shape=(1, 1))
        self.v = self.model.set_variable(var_type="_x", var_name="x_v", shape=(1, 1))
        self.delta = self.model.set_variable(var_type="_x", var_name="x_delta", shape=(1, 1))
        self.a = self.model.set_variable(var_type="_u", var_name="u_acc")
        self.speed_ref = 0
        self.target_speed = self.model.set_variable(var_type="_tvp", var_name="target_speed", shape=(1, 1))

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        lterm, _ = self.cost.cost_function(error, inp)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        _, mterm = self.cost.cost_function(error, inp)
        return mterm

    def set_bounds(self):
        super().set_bounds()
        self.mpc.bounds["lower", "_x", "x_v"] = self.params.v_bounds[0]
        self.mpc.bounds["upper", "_x", "x_v"] = self.params.v_bounds[1]
        self.mpc.bounds["lower", "_u", "u_acc"] = self.params.acc_bounds[0]
        self.mpc.bounds["upper", "_u", "u_acc"] = self.params.acc_bounds[1]

    def _update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref

    def _get_acceleration(self, at: float) -> float:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        try:
            return self.u[2][0]
        except IndexError:
            return self.u[1][0]
