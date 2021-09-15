import math
from dataclasses import dataclass

import casadi
from geometry import translation_angle_from_SE2
from casadi import *
import matplotlib.pyplot as plt
from dg_commons.controllers.mpc.mpc_base_classes import LatMPCBase, LatMPCBAseParam
from typing import Optional, Mapping, Callable, Tuple


__all__ = ["MPCKinCont", "MPCKinContParam"]


def cuberoot(x):
    s = sign(x)
    return s * (s * x) ** (1/3)


def solve(a, b, c, d):
    p = (3 * a * c - b ** 2) / (3 * (a ** 2))
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    summand = -b / (3 * a)
    sol = []

    val1 = -q / 2 - sqrt(q ** 2 / 4 + p ** 3 / 27)
    val2 = -q / 2 + sqrt(q ** 2 / 4 + p ** 3 / 27)

    sol.append(cuberoot(val1) + cuberoot(val2) + summand)

    for i in range(3):
        try:
            val = 1 / 3 * acos((3 * q) / (2 * p) * sqrt(-3 / p)) - 2 * pi * i / 3
            sol.append(2 * sqrt(-p / 3) * cos(val) + summand)
        except:
            sol.append(casadi.inf)
    return sol


@dataclass
class MPCKinContParam(LatMPCBAseParam):
    technique: str = 'linear'
    """ Path Approximation Technique """


class MPCKinCont(LatMPCBase):

    def __init__(self, params: MPCKinContParam = MPCKinContParam()):
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        self.path_var = False
        assert self.params.technique in self.techniques.keys()

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
        self.traj = self.techniques[self.params.technique](self, current_beta)
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

    def _get_linear_traj_func(self, beta):
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

    def _get_quadratic_traj_func(self, beta):
        beta1 = beta
        beta2, beta3 = self.next_beta(beta)
        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)
        q3 = self.path.center_point(beta3)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)
        pos3, angle3 = translation_angle_from_SE2(q3)

        A = np.array([[pos1[0]**2, pos1[0], 1], [pos3[0]**2, pos3[0], 1], [2*pos1[0], 1, 0]])
        b = np.array([[pos1[1]], [pos3[1]], [tan(angle1)]])

        #A = np.array([[pos1[0]**2, pos1[0], 1], [pos2[0]**2, pos2[0], 1], [pos3[0]**2, pos3[0], 1]])
        #b = np.array([[pos1[1]], [pos2[1]], [pos3[1]]])

        res = np.linalg.solve(A, b)
        a, b, c = res[0], res[1], res[2]

        if abs(2*a*pos2[0]) / abs(2*a*pos2[0] + b) < 5*10e-2:
            return self._get_linear_traj_func(beta)

        def quad(x_value):
            return a*x_value**2 + b*x_value + c

        def func(x, y):
            a1 = 2*a**2
            a2 = (3*a*b)
            a3 = (1 - 2 * a * y + b ** 2 + 2 * a * c)
            a4 = (c*b - y * b - x)
            sols = solve(a1, a2, a3, a4)
            dists_list = [power(x_c-x, 2) + power(quad(x_c)-y, 2) for x_c in sols]
            dists = SX(4, 1)
            dists[0, 0] = dists_list[0]
            dists[1, 0] = dists_list[1]
            dists[2, 0] = dists_list[2]
            dists[3, 0] = dists_list[3]

            min_dist = mmin(dists)
            x_sol = casadi.inf
            for sol in sols:
                current_dist = power(sol-x, 2) + power(quad(sol)-y, 2)
                x_sol = if_else(current_dist == min_dist, sol, x_sol)

            return x_sol, quad(x_sol), None

        return func

    def next_beta(self, current_beta):
        along_lane = self.path.along_lane_from_beta(current_beta)
        delta_step = self.current_speed*self.params.t_step*self.params.n_horizon
        along_lane1 = along_lane + delta_step/2
        along_lane2 = along_lane1 + delta_step/2
        return self.path.beta_from_along_lane(along_lane1), self.path.beta_from_along_lane(along_lane2)

    techniques: Mapping[str, Callable[[float], Callable[[float, float], float]]] = \
        {'linear': _get_linear_traj_func, 'quadratic': _get_quadratic_traj_func}
