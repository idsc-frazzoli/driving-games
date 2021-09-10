import math
from dataclasses import dataclass
from typing import Optional

import casadi
import numpy as np

from sim.models.vehicle_structures import VehicleGeometry
from geometry import SE2value, SE2_from_translation_angle, translation_angle_from_SE2
from dg_commons.planning.lanes import DgLanelet
from games import X, U
import do_mpc
from do_mpc.data import save_results
from casadi import *
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import matplotlib as mpl


def kin_euler(x, y, theta, v, delta, s, v_delta, v_s, vehicle_geometry, ts: float):
    state_x = x + cos(theta) * v * ts
    state_y = y + sin(theta) * v * ts
    state_theta = theta + tan(delta) * v * ts / vehicle_geometry.length
    state_v = v + (8 - v) * ts
    state_delta = delta + v_delta * ts
    state_s = s + v_s * ts

    return [state_x, state_y, state_theta, state_v, state_delta, state_s]


def combine_list(list1, list2, alpha1, alpha2):
    zipped_lists = zip(list1, list2)
    return [alpha1 * x + alpha2 * y for (x, y) in zipped_lists]


def rk4(f, param, h: float):
    k1 = f(param)
    k2 = f(combine_list(param, k1, 1, h/2))
    k3 = f(combine_list(param, k2, 1, h/2))
    k4 = f(combine_list(param, k3, 1, h))
    k = combine_list(combine_list(combine_list(k1, k2, 1, 2), k3, 1, 2), k4, 1, 1)
    k = [h/6 * i for i in k]
    return combine_list(param, k, 1, 1)


def kin_rk4(x, y, theta, v, delta, s, v_delta, v_s, vehicle_geometry, ts: float):
    def f(param):
        return_val = []
        return_val.append(cos(param[2]) * param[3])
        return_val.append(sin(param[2]) * param[3])
        return_val.append(tan(param[4]) * param[3] / vehicle_geometry.length)
        return_val.append(0)
        return_val.append(v_delta)
        return_val.append(v_s)
        return return_val

    parameters = [x, y, theta, v, delta, s]
    result = rk4(f, parameters, ts)

    return result


discretizations = {'Kinematic Euler': kin_euler, 'Kinematic RK4': kin_rk4}
