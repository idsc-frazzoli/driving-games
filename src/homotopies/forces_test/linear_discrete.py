import numpy as np
import matplotlib.pyplot as plt
import forcespro
import forcespro.nlp
import casadi

"""
Simple MPC - double integrator example for use with FORCESPRO
with rate constraints on du

 min   xN'*P*xN + sum_{i=1}^{N-1} xi'*Q*xi + ui'*R*ui
xi,ui
     s.t. x1 = x
       x_i+1 = A*xi + B*ui  for i = 1...N-1
       xmin  <= xi <= xmax   for i = 1...N
       umin  <= ui <= umax   for i = 1...N
       dumin <= u{i+1} - ui <= dumax for i= 1...N-1

 and P is solution of Ricatti eqn. from LQR problem

(c) Embotech AG, Zurich, Switzerland, 2013-2021
"""


# Model Definition
# ----------------
def obj(z):
    return z[1] * R * z[1] + casadi.horzcat(z[2], z[3]) @ Q @ casadi.vertcat(z[2], z[3])


def objN(z):
    return z[1] * R * z[1] + casadi.horzcat(z[2], z[3]) @ P @ casadi.vertcat(z[2], z[3])


def dynamics(z):
    return casadi.vertcat(z[0] + z[1],
                          casadi.dot(A[0, :], casadi.vertcat(z[2], z[3])) + B[0, :] * z[1],
                          casadi.dot(A[1, :], casadi.vertcat(z[2], z[3])) + B[1, :] * z[1])

def c_dynamics(x,u):
    return casadi.vertcat(u[0],
                          A[0, 0]+ A[0,1] * x[2]/dT + B[0, :] * x[0]/dT,
                          B[1, :] * x[0]/dT)


# system
A = np.array([[1.1, 1], [0, 1]])
B = np.array([[1], [0.5]])
nx, nu = np.shape(B)

# MPC setup
N = 10
Q = np.eye(nx)
R = np.eye(nu)
P = 10 * Q
umin = -10
umax = 10
absrate = 2
dumin = -absrate
dumax = absrate
xmin = np.array([-50, -50])
xmax = np.array([50, 50])
dT=0.1
# FORCESPRO multistage form
# assume variable ordering zi = [u{i+1}-ui; ui; xi] for i=1...N

# dimensions
model = forcespro.nlp.SymbolicModel(10)  # horizon length
model.nvar = 4  # number of variables
model.neq = 3  # number of equality constraints

# objective
model.objective = obj
model.objectiveN = objN
# equalities
model.eq = dynamics
# model.continuous_dynamics = c_dynamics
model.E = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)

# initial state
model.xinitidx = [2, 3]

# Bounds
# In the first stage, we have parametric bounds on the inputs.
model.lbidx[0] = range(0, 2)
model.ubidx[0] = range(0, 2)
# In the following stages, all stage variables (inputs and states) are bounded.
for i in range(1, N):
    model.lbidx[i] = range(0, 4)
    model.ubidx[i] = range(0, 4)

model.intidx = [0]
# Generate FORCESPRO solver
# -------------------------

# set options
options = forcespro.CodeOptions('L_test')
options.solvemethod = "PDIP_NLP"
options.printlevel = 0
options.overwrite = 1
options.nlp.bfgs_init = None
options.forcenonconvex = 1

options.nlp.integrator.type = 'ERK4'
options.nlp.integrator.Ts = dT
options.nlp.integrator.nodes = 10
# generate code
solver = model.generate_solver(options)
solver.help()
# help(solver)
# Run simulation
# --------------

x1 = [-40, 20]
kmax = 30
x = np.zeros((2, kmax + 1))
x[:, 0] = x1
u = np.zeros((1, kmax))
du = np.zeros((1, kmax))
problem = {}

solvetime = []
iters = []

for k in range(kmax):
    problem["xinit"] = x[:, k]
    problem["lb{:02d}".format(1)] = [dumin, umin]  # matlab-based index
    problem["ub{:02d}".format(1)] = [dumax, umax]
    for s in range(1, N):
        problem["lb{:02d}".format(s + 1)] = np.concatenate([[dumin, umin], xmin])
        problem["ub{:02d}".format(s + 1)] = np.concatenate([[dumax, umax], xmax])
    problem["x0"] = np.transpose(np.tile(np.concatenate([np.zeros(2), x[:, k]]), (1, model.N)))
    # call the solver
    solverout, exitflag, info = solver.solve(problem)
    # assert exitflag == 1, "Some problem in solver"

    du[:, k] = solverout["x01"][0]
    u[:, k] = solverout["x01"][1]
    # solvetime.append(info.minlpSolveTime)
    # iters.append(info.it)
    print("exitflag",exitflag)
    print(solverout["x01"])
    print(model.eq(solverout["x01"]))
    update = model.eq(np.concatenate([du[:, k], u[:, k], x[:, k]])).full().reshape(3, )
    x[:, k + 1] = update[1:]
    if k + 1 < kmax:
        u[:, k + 1] = update[0]

# Plot results
# ------------

fig = plt.gcf()
plt.subplot(3, 1, 1)
plt.grid('both')
plt.title('states')
plt.plot([1, kmax], [xmin[0], xmin[0]], 'r--')
plt.plot([1, kmax], [xmax[0], xmax[0]], 'r--')
plt.ylim(1.1 * np.array([xmin[0], xmax[0]]))
plt.step(range(1, kmax + 1), x[0, range(1, kmax + 1)])
plt.step(range(1, kmax + 1), x[1, range(1, kmax + 1)])

plt.subplot(3, 1, 2)
plt.grid('both')
plt.title('input')
plt.plot([1, kmax], [umin, umin], 'r--')
plt.plot([1, kmax], [umax, umax], 'r--')
plt.ylim(1.1 * np.array([umin, umax]))
plt.step(range(1, kmax + 1), u[0, range(0, kmax)])

plt.subplot(3, 1, 3)
plt.grid('both')
plt.title('input')
plt.plot([1, kmax], [absrate, absrate], 'r--')
plt.plot([1, kmax], [-absrate, -absrate], 'r--')
plt.ylim(1.1 * np.array([-absrate, absrate]))
plt.step(range(1, kmax + 1), du[0, range(0, kmax)])

plt.show()
