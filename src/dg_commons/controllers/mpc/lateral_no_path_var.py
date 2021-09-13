from dataclasses import dataclass
from geometry import translation_angle_from_SE2
from casadi import *
import matplotlib.pyplot as plt
from dg_commons.controllers.mpc.mpc_base_classes import LatMPCBase, LatMPCBAseParam

__all__ = ["MPCKinCont", "MPCKinContParam"]


@dataclass
class MPCKinContParam(LatMPCBAseParam):
    pass


class MPCKinCont(LatMPCBase):

    def __init__(self, params: MPCKinContParam = MPCKinContParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        self.path_var = False

        # Set right right hand side of differential equation for x, y, theta, v, and delta
        self.model.set_rhs('state_x', cos(self.theta) * self.v)
        self.model.set_rhs('state_y', sin(self.theta) * self.v)
        self.model.set_rhs('theta', tan(self.delta) * self.v / self.vehicle_geometry.length)
        self.model.set_rhs('v', casadi.SX(0))
        self.model.set_rhs('delta', self.v_delta)

        self.model.setup()

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        return self.params.state_mult * ((target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2) + \
                self.params.input_mult * self.v_delta ** 2

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        return (target_x - self.state_x) ** 2 + (target_y - self.state_y) ** 2

    def compute_targets(self, current_beta):
        self.traj = self.get_traj_func(current_beta)
        return self.traj(self.state_x, self.state_y)

    def set_bounds(self):
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]

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

    def get_traj_func(self, beta):
        beta1, beta2 = beta, beta + 1
        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)

        x1, y1 = pos1[0], pos1[1]
        x2, y2 = pos2[0], pos2[1]
        plt.xlim(-10, 10), plt.ylim(-10, 10)
        plt.plot(x1, y1, x2, y2, marker='o')
        plt.show()

        vertical_line = False
        if pos1[0] == pos2[0] and pos1[1] != pos2[2]:
            vertical_line = True
            res = [pos1[0], 1 if pos1[1] < pos2[1] else -1]
        else:
            m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
            b = pos1[1] - m * pos1[0]
            res = [m, b]

        def func(x, y):
            if vertical_line:
                return res[0], y, res[1] * pi / 2
            else:
                x_val = (x + res[0] * (y - res[1])) / (1 + res[0] ** 2)
                y_val = (res[0] ** 2 * y + res[0] * x + res[1]) / (1 + res[0] ** 2)
                angle = atan(res[0])
                angle = angle + pi if (res[0] > 0 and pos2[1] < pos1[1]) or (
                            res[0] < 0 and pos1[1] < pos2[1]) else angle
                return x_val, y_val, angle

        return func
