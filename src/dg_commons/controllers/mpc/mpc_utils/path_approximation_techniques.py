from casadi import *


def linear_param(pos1, angle1, pos2, angle2, pos3, angle3):
    vertical_line = False
    if abs(pos1[0] - pos2[0]) < 10e-8:
        vertical_line = True
        res = [pos1[0], sign(pos2[1]-pos1[1])]

        def func(x):
            return None
    else:
        m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
        b = pos1[1] - m * pos1[0]
        res = [m, b]

        def func(x):
            return res[0] * x + res[1]

    return res, func, vertical_line


def cubic_param(pos1, angle1, pos2, angle2, pos3, angle3):
    vertical_line = False
    if abs(pos1[0] - pos2[0]) < 10e-8 and abs(pos2[0] - pos3[0]) < 10e-8:
        vertical_line = True
        res = [pos1[0], sign(pos2[1]-pos1[1])]

        def func(x):
            return None
    else:
        A = np.array([[pos1[0] ** 3, pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 3, pos3[0] ** 2, pos3[0], 1],
                      [3 * pos1[0] ** 2, 2 * pos1[0], 1, 0], [3 * pos3[0] ** 2, 2 * pos3[0], 1, 0]])
        b = np.array([[pos1[1]], [pos3[1]], [tan(angle1)], [tan(angle3)]])
        res = np.linalg.solve(A, b)

        def func(x):
            return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x + res[3]

    return res, func, vertical_line


def quadratic_param(pos1, angle1, pos2, angle2, pos3, angle3, pref='var3'):
    vertical_line = False
    if abs(pos1[0] - pos2[0]) < 10e-8 and abs(pos2[0] - pos3[0]) < 10e-8:
        vertical_line = True
        res = [pos1[0], sign(pos2[1]-pos1[1])]

        def func(x):
            return None
    else:
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

    return res, func, vertical_line


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
