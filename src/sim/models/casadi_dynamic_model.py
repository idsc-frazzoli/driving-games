from geometry import SO2_from_angle
from sim.models.vehicle import VehicleGeometry
from sim.models import Pacejka4p
from casadi import *
from sim.models.utils import G, kmh2ms
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn

# states = VehicleStateDyn(x=MX.sym('x'),
#                          y=MX.sym('y'),
#                          theta=MX.sym('theta'),
#                          vx=MX.sym('dx'),
#                          vy=MX.sym('dy'),
#                          dtheta=MX.sym('dtheta'),
#                          delta=MX.sym('delta')
#                          )
#
# inputs = VehicleCommands(acc=MX.sym('acc'),
#                          ddelta=MX.sym('ddelta')
#                          )

x0_p1 = VehicleStateDyn(x=-37, y=-8, theta=0.05, vx=kmh2ms(40), delta=0)
P1 = VehicleModelDyn.default_car(x0_p1)
# Control
inputs = SX.sym("u", 2)
acc = inputs[0]
ddelta = inputs[1]

# State
states = SX.sym("x", 7)
x = states[0]  # position
y = states[1]
theta = states[2]
dx = states[3]
dy = states[4]
dtheta = states[5]
delta = states[6]

# ODE right hand side

costh = cos(theta)
sinth = sin(theta)

xdot = dx * costh - dy * sinth
ydot = dx * sinth + dy * costh
theta_dot = dtheta

# vertical forces
m = P1.vg.m
load_transfer = P1.vg.h_cog * acc
F1_n = -m * (G * P1.vg.lr - load_transfer) / (P1.vg.lr + P1.vg.lr)
F2_n = -m * (G * P1.vg.lf + load_transfer) / (P1.vg.lr + P1.vg.lr)

Facc1 = 0
Facc2 = m * acc

# front wheel forces (assumes no longitudinal force, rear traction)
vel_1_tyre = []
vel_1_tyre.append(dx * cos(delta) - (dy + P1.vg.lf * dtheta) * sin(delta))
vel_1_tyre.append(dx * sin(delta) + (dy + P1.vg.lf * dtheta) * cos(delta))
pacejka_front = P1.pacejka_front
pacejka_rear = P1.pacejka_rear

slip_angle_1 = atan(vel_1_tyre[1] / vel_1_tyre[0])

F1y_tyre = pacejka_front.evaluate(slip_angle_1) * F1_n
Facc1_sat = Facc1 * sqrt(1 - (F1y_tyre / (F1_n * pacejka_front.D)) ** 2)
#F1 = rot_delta.T @ np.array([Facc1_sat, F1y_tyre])
F1 = []
F1.append(Facc1_sat * cos(delta) + F1y_tyre * sin(delta))
F1.append(- Facc1_sat * sin(delta) + F1y_tyre * cos(delta))

vel_2 = [dx, dy - P1.vg.lr * dtheta]
slip_angle_2 = atan(vel_2[1] / vel_2[0])
# Back wheel forces (implicit assumption motor on the back)
F2y = pacejka_rear.evaluate(slip_angle_2) * F2_n

# Saturate longitudinal acceleration based on the used lateral one
Facc2_sat = Facc2 * sqrt(1 - (F2y / (F2_n * pacejka_rear.D)) ** 2)

# Drag Force
# longitudinal acceleration
acc_x = (F1[0] + Facc2_sat + m * dtheta * dy) / m
acc_y = (F1[1] + F2y - m * dtheta * dx) / m
# yaw acceleration
ddtheta = (F1[1] * P1.vg.lf - F2y * P1.vg.lr) / P1.vg.Iz

xdotdot = acc_x
ydotdot = acc_y
thetadotdot = ddtheta

deltadot = ddelta
states_dot = vertcat(xdot, ydot, theta_dot, xdotdot, ydotdot, thetadotdot, deltadot)

# ODE right hand side function
f = Function('f', [states, inputs], [xdot])

# # Integrate with Explicit Euler over 0.2 seconds
# dt = 0.01  # Time step
# xj = x
# for j in range(20):
#     fj = f(xj, inputs)
#     xj += dt * fj
#
# # Discrete time dynamics function
# F = Function('F', [states, inputs], [xj])
#
# # Number of control segments
# nu = 50
#
# # Control for all segments
# U = MX.sym("U", nu)
#
# # Initial conditions
# X0 = MX([0, 0, 1])
#
# # Integrate over all intervals
# X = X0
# for k in range(nu):
#     X = F(X, U[k])
#
# # Objective function and constraints
# J = mtimes(U.T, U)  # u'*u in Matlab
# G = X[0:2]  # x(1:2) in Matlab
#
# # NLP
# nlp = {'x': U, 'f': J, 'g': G}
#
# # Allocate an NLP solver
# opts = {"ipopt.tol": 1e-10, "expand": True}
# solver = nlpsol("solver", "ipopt", nlp, opts)
# arg = {}
#
# # Bounds on u and initial condition
# arg["lbx"] = -0.5
# arg["ubx"] = 0.5
# arg["x0"] = 0.4
#
# # Bounds on g
# arg["lbg"] = [10, 0]
# arg["ubg"] = [10, 0]
#
# # Solve the problem
# res = solver(**arg)
#
# # Get the solution
# plt.plot(res["x"])
# plt.plot(res["lam_x"])
# plt.savefig("mygraph.png")
#
