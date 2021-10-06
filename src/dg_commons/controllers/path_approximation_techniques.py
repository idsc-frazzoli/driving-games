from casadi import *
from typing import Callable, Mapping


def vertical_line_param(pos1, pos2):
    vertical = True
    return pos1[0], sign(pos2[1] - pos1[1]), vertical


def linear_param(pos1, angle1, pos2, angle2, pos3, angle3):
    vertical = False
    if abs(pos1[0] - pos2[0]) < 10e-8:
        return vertical_line_param(pos1, pos2)

    m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
    b = pos1[1] - m * pos1[0]
    angle = atan2(pos2[1] - pos1[1], pos2[0] - pos1[0])  # returns a value in [-pi, pi]
    angle = 2 * pi + angle if angle < 0 else angle  # returns a value in [0, 2pi]

    return m, b, angle, vertical


def quadratic_param(pos1, angle1, pos2, angle2, pos3, angle3, pref='var3'):
    vertical = False
    if abs(pos1[0] - pos2[0]) < 10e-8 and abs(pos2[0] - pos3[0]) < 10e-8:
        return vertical_line_param(pos1, pos2)

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

    return res[0][0], res[1][0], res[2][0], vertical


def cubic_param(pos1, angle1, pos2, angle2, pos3, angle3):
    vertical = False
    '''if abs(pos1[0] - pos2[0]) < 10e-8 and abs(pos2[0] - pos3[0]) < 10e-8:
        return vertical_line_param(pos1, pos2)'''

    A = np.array([[pos1[0] ** 3, pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 3, pos3[0] ** 2, pos3[0], 1],
                  [3 * pos1[0] ** 2, 2 * pos1[0], 1, 0], [3 * pos3[0] ** 2, 2 * pos3[0], 1, 0]])
    b = np.array([[pos1[1]], [pos3[1]], [tan(angle1)], [tan(angle3)]])
    res = np.linalg.lstsq(A, b)[0]
    return res[0][0], res[1][0], res[2][0], res[3][0], vertical


def vertical_line(x, s):
    vertical_line = True
    angle = pi / 2 * s
    angle = 2*pi + angle if angle < 0 else angle  # returns a value in [0, 2pi]
    res = [x, s, angle]

    def func(x):
        return None

    def closest_point(pos):
        x_val = res[0]
        y_val = pos[1]
        return x_val, y_val

    return res, func, vertical_line, closest_point


def linear(m, b, angle):
    vertical = False

    res = [m, b, angle]

    def func(x):
        return res[0] * x + res[1]

    def closest_point(pos):
        x_val = (pos[0] + res[0] * (pos[1] - res[1])) / (1 + res[0] ** 2)
        y_val = (res[0] ** 2 * pos[1] + res[0] * pos[0] + res[1]) / (1 + res[0] ** 2)
        return [x_val, y_val]

    return res, func, vertical, closest_point


def quadratic(a, b, c):
    res = [a, b, c]

    def func(x):
        return a * x ** 2 + b * x + c

    def closest_point(pos):
        a1 = 2 * a ** 2
        a2 = (3 * a * b)
        a3 = (1 - 2 * a * pos[1] + b ** 2 + 2 * a * c)
        a4 = (c * b - pos[1] * b - pos[0])
        sols = solve_quadratic(a1, a2, a3, a4)
        dists_list = [power(x_c - pos[0], 2) + power(func(x_c) - pos[1], 2) for x_c in sols]
        dists = SX(4, 1)
        dists[0, 0] = dists_list[0]
        dists[1, 0] = dists_list[1]
        dists[2, 0] = dists_list[2]
        dists[3, 0] = dists_list[3]

        min_dist = mmin(dists)
        x_sol = casadi.inf
        for sol in sols:
            current_dist = power(sol - pos[0], 2) + power(func(sol) - pos[1], 2)
            x_sol = if_else(current_dist == min_dist, sol, x_sol)

        return [x_sol, func(x_sol)]

    return res, func, vertical_line, closest_point


def cubic(a, b, c, d):
    vertical = False
    res = [a, b, c, d]

    def func(x):
        return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x + res[3]

    return res, func, vertical, None


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

