import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp

from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()


class MIPModelParams:
    Nstages = 15
    nobs = 1  # number of obstacles considered
    nx = 5
    nb = 4 * nobs  # binary control input
    nc = 2  # continuous control input
    nu = nb + nc
    nz = nx + nu
    npar = 3 * nobs + 2  # runtime parameter: obstacle states and target pos
    nh = 5 * nobs  # inequality constraint: binary constraints and region constraints


params = MIPModelParams


def continuous_dynamics(x, u):
    """Defines dynamics of the vehicle, i.e. equality constraints.
    state x = [xpos, ypos, theta, v, delta]
    input u = [binary_variables, acc,v_delta]
    """
    u = u[params.nb:]
    xpos = x[0]
    ypos = x[1]
    theta = x[2]
    v = x[3]
    delta = x[4]
    acc = u[0]
    v_delta = u[1]
    dtheta = v * casadi.tan(delta) / vehicle_geometry.length
    vy = dtheta * vehicle_geometry.lr
    dxpos = v * casadi.cos(theta) - vy * casadi.sin(theta)
    dypos = v * casadi.sin(theta) + vy * casadi.cos(theta)

    return casadi.vertcat(dxpos,  # dx
                          dypos,  # dy
                          dtheta,  # dtheta
                          acc,  # dv
                          v_delta,  # ddelta
                          )


def obj(z, p):
    """Least square costs on deviating from the path and on the inputs
    z = [binary_variables, acc,v_delta,xpos, ypos, theta, v, delta]
    p = [pred_state, target_pos]
    current_target = point on path that is to be headed for
    """
    z = z[params.nb:]
    current_target = p[3 * params.nobs:]
    return (200.0 * (z[2] - current_target[0]) ** 2  # costs on deviating on the
            #                                              path in x-direction
            + 200.0 * (z[3] - current_target[1]) ** 2  # costs on deviating on the
            #                                               path in y-direction
            + 1 * z[0] ** 2  # penalty on input acc_left
            + 1 * z[1] ** 2)  # penalty on input acc_right


def objN(z, p):
    """Increased least square costs for last stage on deviating from the path and
    on the inputs
    z = [binary_variables, acc,v_delta,xpos, ypos, theta, v, delta]
    p = [pred_state, target_pos]
    current_target = point on path that is to be headed for
    """
    z = z[params.nb:]
    current_target = p[3 * params.nobs:]
    return (200.0 * (z[2] - current_target[0]) ** 2  # costs on deviating on the
            #                                              path in x-direction
            + 200.0 * (z[3] - current_target[1]) ** 2  # costs on deviating on the
            #                                               path in y-direction
            + 2 * z[0] ** 2  # penalty on acc_left
            + 2 * z[1] ** 2)  # penalty on acc_right


def set_bounds():
    '''z = [binary_variables, acc,v_delta,xpos, ypos, theta, v, delta]
    this function returns lower and upper bounds for continuous variables
    '''
    v_delta_bounds = np.array([-vehicle_params.ddelta_max, vehicle_params.ddelta_max])
    delta_bounds = np.array([-vehicle_params.default_car().delta_max, vehicle_params.default_car().delta_max])
    acc_bounds = np.array(vehicle_params.acc_limits)
    v_bounds = np.array(vehicle_params.vx_limits)
    default_bounds = np.array([-np.inf, np.inf])
    all_bounds = np.array([acc_bounds,
                           v_delta_bounds,
                           default_bounds,
                           default_bounds,
                           default_bounds,
                           v_bounds,
                           delta_bounds])
    return all_bounds


def binary_ineq(z):
    """z = [binary_variables, acc,v_delta,xpos, ypos, theta, v, delta]
    this function returns equality constraints for binary variables
    """
    ub = z[:params.nb]
    return casadi.sum1(ub)


def region_ineq(z, p):
    """z = [binary_variables, acc,v_delta, xpos, ypos, theta, v, delta]
    p = [pred_state, target_pos]
    this function returns equality constraints for regions formed by obstacle
    """
    ub = z[:params.nb]
    xpos = z[params.nu]
    ypos = z[params.nu + 1]
    xpos_obs = p[0]
    ypos_obs = p[1]
    return casadi.vertcat(
        ub[0] * (ypos_obs - 2 - ypos),
        ub[1] * (xpos_obs - 1 - xpos),
        ub[2] * (ypos - ypos_obs - 2),
        ub[3] * (xpos - xpos_obs - 1)
    )


def get_ineq(z, p):
    ineq1 = binary_ineq(z)
    ineq2 = region_ineq(z, p)
    return casadi.vertcat(ineq1, ineq2)


def get_obs_pred(s_obs):
    """this function takes as input the current state of an obstacle,
    outputs its predicted position and orientation at each state.
    the output will be provided to the solver as runtime parameters"""
    x_obs = s_obs.x
    y_obs = s_obs.y
    theta_obs = s_obs.theta
    pred = np.zeros([params.Nstages, 3])
    for i in range(params.Nstages):
        pred[i, :] = np.array([x_obs, y_obs, theta_obs])
    return pred


def create_model():
    # Model Definition
    # ----------------
    # Problem dimensions
    model = forcespro.nlp.SymbolicModel(params.Nstages)
    model.nvar = params.nz  # number of variables
    model.neq = params.nx  # number of equality constraints
    model.nh = params.nh  # number of nonlinear inequality constraints
    model.npar = params.npar  # number of runtime parameters--- target position

    # Objective function
    model.objective = obj
    model.objectiveN = objN  # increased costs for the last stage

    # Differential equality constraints
    model.continuous_dynamics = continuous_dynamics
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((params.nx, params.nu)), np.eye(params.nx)], axis=1)

    # Discrete equality constraints, formulated as inequality constraints (i.e. Ax<=b & Ax>=b)
    model.ineq = get_ineq
    model.hu = np.concatenate([1.5 * np.ones(params.nobs),  # upper bound for binary constraints
                               np.inf * np.ones(4 * params.nobs)  # upper bound for region constraints
                               ])
    model.hl = np.concatenate([0.5 * np.ones(params.nobs),  # lower bound for binary constraints
                               np.zeros(4 * params.nobs)  # lower bound for region constraints
                               ])

    # Inequality constraints
    # # In the first stage, we have parametric bounds on the inputs.
    model.lbidx[0] = range(0, params.nu)
    model.ubidx[0] = range(0, params.nu)
    # # In the following stages, all stage variables (inputs and states) are bounded.
    for i in range(1, model.N):
        model.lbidx[i] = range(0, params.nz)
        model.ubidx[i] = range(0, params.nz)
    # # Integer indices
    model.intidx = range(0, params.nb)

    # Initial condition on vehicle states x, use this to specify on which variables initial conditions are imposed
    model.xinitidx = range(params.nu, params.nz)
    return model


def generate_pathplanner():
    """Generates and returns a FORCESPRO solver that calculates a path based on
    constraints and dynamics while minimizing an objective function
    """
    model = create_model()
    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
    codeoptions.maxit = 2000  # Maximum number of iterations
    codeoptions.printlevel = 2  # Use printlevel = 2 to print progress (but
    #                             not for timings)
    codeoptions.optlevel = 0  # 0 no optimization, 1 optimize for size,
    #                             2 optimize for speed, 3 optimize for size & speed
    codeoptions.cleanup = True
    codeoptions.timing = 1

    codeoptions.noVariableElimination = 1  #To prevent state variables elimination at stage 0
    # codeoptions.nlp.hessian_approximation = 'bfgs'
    # codeoptions.nlp.bfgs_init = 1 * np.identity(8)

    codeoptions.nlp.integrator.type = 'IRK2'
    codeoptions.nlp.integrator.Ts = 0.05
    codeoptions.nlp.integrator.nodes = 20
    codeoptions.minlp.int_guess = 1
    codeoptions.minlp.round_root = 1
    codeoptions.minlp.int_guess_stage_vars = [0]
    # change this to your server or leave uncommented for using the
    # standard embotech server at https://forces.embotech.com
    # codeoptions.server = 'https://forces.embotech.com'

    # Creates code for symbolic model formulation given above, then contacts
    # server to generate new solver
    outputs = [('ub', range(0, model.N), range(0, params.nb)),
               ('uc', range(0, model.N), range(params.nb, params.nu)),
               ('x', range(0, model.N), range(params.nu, params.nz))]
    solver = model.generate_solver(options=codeoptions, outputs=outputs)

    return model, solver
