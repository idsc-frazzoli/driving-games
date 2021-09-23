import numpy as np
from casadi import *
from dg_commons.utils import SemiDef
from dataclasses import dataclass
from typing import Union

def linear_param(pos1, angle1, pos2, angle2, pos3, angle3):
    m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
    b = pos1[1] - m * pos1[0]
    res = [m, b]

    def func(x):
        return res[0] * x + res[1]

    return res, func


def cubic_param(pos1, angle1, pos2, angle2, pos3, angle3):

    A = np.array([[pos1[0] ** 3, pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 3, pos3[0] ** 2, pos3[0], 1],
                  [3 * pos1[0] ** 2, 2 * pos1[0], 1, 0], [3 * pos3[0] ** 2, 2 * pos3[0], 1, 0]])
    b = np.array([[pos1[1]], [pos3[1]], [tan(angle1)], [tan(angle3)]])
    res = np.linalg.solve(A, b)

    def func(x):
        return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x + res[3]

    return res, func


def quadratic_param(pos1, angle1, pos2, angle2, pos3, angle3, pref='var3'):

    if pref == 'var1':
        A = np.array([[pos1[0] ** 2, pos1[0], 1], [pos2[0] ** 2, pos2[0], 1], [pos3[0] ** 2, pos3[0], 1]])
        b = np.array([[pos1[1]], [pos2[1]], [pos3[1]]])
    elif pref == 'var2':
        A = np.array([[pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 2, pos3[0], 1], [2 * pos1[0], 1, 0]])
        b = np.array([[pos1[1]], [pos2[1]], [tan(angle1)]])
    elif pref == 'var3':
        A = np.array([[pos1[0] ** 2, pos1[0], 1], [pos2[0] ** 2, pos2[0], 1], [pos3[0] ** 2, pos3[0], 1],
                      [2 * pos1[0], 1, 0], [2 * pos2[0], 1, 0], [2 * pos3[0], 1, 0]])
        b = np.array([[pos1[1]], [pos2[1]], [pos3[1]], [tan(angle1)], [tan(angle2)], [tan(angle3)]])

    res = np.linalg.lstsq(A, b)[0]

    def func(x):
        return res[0] * x ** 2 + res[1] * x + res[2]

    return res, func


def cuberoot(x):
    s = sign(x)
    return s * (s * x) ** (1/3)


def solve_quadratic(a, b, c, d):
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


def mat_mul(X, Y):


    result = [([0]*len(Y[0]))*len(X)]

    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]


class Empty:
    pass


@dataclass
class quadratic_params:
    q: SemiDef
    r: SemiDef


def quadratic_cost(x, u, quad_params):
    r = SX(quad_params.r.matrix)
    q = SX(quad_params.q.matrix)

    dim_x = len(x)
    dim_u = len(u)
    helper1 = GenSX_zeros(dim_x)
    helper2 = GenSX_zeros(dim_u)

    for i in range(dim_x):
        helper1[i] = x[i]

    for i in range(dim_u):
        helper2[i] = u[i]

    return bilin(q, helper1, helper1) + bilin(r, helper2, helper2), bilin(q, helper1, helper1)


CostParameters = Union[Empty, quadratic_params]
costs = {"quadratic": quadratic_cost}


