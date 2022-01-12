import forcespro
import forcespro.nlp
import numpy as np
import casadi
import matplotlib.pyplot as plt

"""
here we test a simple MIP problem
states: xpos, ypos, theta
inputs: v(continuous), turn(-1,0,1)
parameter: target pos
z=[v, turn, xpos, ypos, theta]
discrete dynamics:
xpos(i+1)=xpos(i)+T*v*cos(theta)
ypos(i+1)=ypos(i)+T*v*sin(theta)
theta(i+1) =theta(i)+turn*pi/2

continuous dynamics:
dx=v*cos(theta)
dy=v*sin(theta)
dtheta=turn*pi/2/T

obj:distance between current pos and target pos
"""


def obj(z):
    """z=[v, turn, xpos, ypos, theta]"""
    return ((z[2] - 10) ** 2
            + (z[3] - 10) ** 2)


def objN(z):
    """z=[v, turn, xpos, ypos, theta]"""
    return ((z[2] - 10) ** 2
            + (z[3] - 10) ** 2)


def discrete_dynamics(z):
    """z=[v, turn, xpos, ypos, theta]
discrete dynamics:
xpos(i+1)=xpos(i)+T*v*cos(theta)
ypos(i+1)=ypos(i)+T*v*sin(theta)
theta(i+1) =theta(i)+turn*pi/2"""
    u = z[0:2]
    x = z[2:5]
    xpos_1 = x[0] + dT * u[0] * casadi.cos(x[2])
    ypos_1 = x[1] + dT * u[0] * casadi.sin(x[2])
    theta_1 = x[2] + u[1] * (np.pi / 2)
    return casadi.vertcat(xpos_1,
                          ypos_1,
                          theta_1)


# system
nx = 3
nu = 2

# MPC setup
N = 10
dT = 0.1
umin = np.array([-1., -1])
umax = np.array([1., 1])
xmin = np.array([-15, -15, -np.inf])
xmax = np.array([15, 15, np.inf])

model = forcespro.nlp.SymbolicModel(N)

model.nvar = nx + nu  # number of stage variables
model.neq = nx  # number of equality constraints
model.nh = 0  # number of nonlinear inequality constraints
model.npar = 0  # number of runtime parameters

model.objective = obj  # eval_obj is a Python function
model.objectiveN = objN

model.eq = discrete_dynamics  # handle to inter-stage function
model.E = np.concatenate([np.zeros((nx, nu)), np.eye(nx)], axis=1)  # selection matrix

# In the first stage, we have parametric bounds on the inputs.
model.lbidx[0] = range(0, nu)
model.ubidx[0] = range(0, nu)
# In the following stages, all stage variables (inputs and states) are bounded.
for i in range(1, N):
    model.lbidx[i] = range(0, model.nvar)
    model.ubidx[i] = range(0, model.nvar)
# model.intidx = [1]
model.xinitidx = range(nu, model.nvar)
# Get the default solver options
options = forcespro.CodeOptions('NL_test')
options.solvemethod = "PDIP_NLP"
options.printlevel = 2
options.overwrite = 1
options.nlp.bfgs_init = None

# codeoptions.nlp.integrator.type = 'IRK2'
# codeoptions.nlp.integrator.Ts = dT
# codeoptions.nlp.integrator.nodes = 2
# Specify maximum number of threads to parallelize minlp search
# codeoptions.minlp.max_num_threads = 8


# Generate solver for previously initialized model
solver = model.generate_solver(options)
# solver = forcespro.nlp.Solver.from_directory("./mpc_test")
solver.help()

problem = {}
problem["lb{:02d}".format(1)] = umin  # matlab-based index
problem["ub{:02d}".format(1)] = umax
for s in range(1, N):
    problem["lb{:02d}".format(s + 1)] = np.concatenate([umin, xmin])
    problem["ub{:02d}".format(s + 1)] = np.concatenate([umax, xmax])

problem["x0"] = np.tile(np.zeros((model.nvar, 1)), (N, 1))
problem["xinit"] = np.array([0,0,0]).reshape(3,1)
# problem["all_parameters"] = np.tile(np.array([10,10]), (N,1))
# FORCESPRO integer search will run on 2 thread
# problem["parallelStrategy"] = 0  # Default value
# problem["numThreadsBnB"] = 2

sol, exitflag, info = solver.solve(problem)
