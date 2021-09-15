from casadi import *


def linear_param(pos1, angle1, pos2, angle2, pos3, angle3):

    m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
    b = pos1[1] - m * pos1[0]
    return [m, b]


def cubic_param(pos1, angle1, pos2, angle2, pos3, angle3):

    A = np.array([[pos1[0] ** 3, pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 3, pos3[0] ** 2, pos3[0], 1],
                  [3 * pos1[0] ** 2, 2 * pos1[0], 1, 0], [3 * pos3[0] ** 2, 2 * pos3[0], 1, 0]])
    b = np.array([[pos1[1]], [pos3[1]], [tan(angle1)], [tan(angle3)]])

    return np.linalg.solve(A, b)


def quadratic_param(pos1, angle1, pos2, angle2, pos3, angle3, pref='Default'):

    if pref == 'var1':
        A = np.array([[pos1[0] ** 2, pos1[0], 1], [pos2[0] ** 2, pos2[0], 1], [pos3[0] ** 2, pos3[0], 1]])
        b = np.array([[pos1[1]], [pos2[1]], [pos3[1]]])
    else:
        A = np.array([[pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 2, pos3[0], 1], [2 * pos1[0], 1, 0]])
        b = np.array([[pos1[1]], [pos2[1]], [tan(angle1)]])

    return np.linalg.solve(A, b)


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
