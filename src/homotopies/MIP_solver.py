import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp

from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()

def continuous_dynamics(x, u):
    """Defines dynamics of the vehicle, i.e. equality constraints.
    state x = [xpos, ypos, theta, v, delta]
    input u = [acc,v_delta]
    """
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
                          v_delta,  #ddelta
                          )


def obj(z, current_target):
    """Least square costs on deviating from the path and on the inputs
    z = [acc,v_delta,xpos, ypos, theta, v, delta]
    current_target = point on path that is to be headed for
    """
    return (200.0 * (z[2] - current_target[0]) ** 2  # costs on deviating on the
            #                                              path in x-direction
            + 200.0 * (z[3] - current_target[1]) ** 2  # costs on deviating on the
            #                                               path in y-direction
            + 1 * z[0] ** 2  # penalty on input acc_left
            + 1 * z[1] ** 2)  # penalty on input acc_right


def objN(z, current_target):
    """Increased least square costs for last stage on deviating from the path and
    on the inputs
    z = [acc,v_delta,xpos, ypos, theta, v, delta]
    current_target = point on path that is to be headed for
    """
    return (200.0 * (z[2] - current_target[0]) ** 2  # costs on deviating on the
            #                                              path in x-direction
            + 200.0 * (z[3] - current_target[1]) ** 2  # costs on deviating on the
            #                                               path in y-direction
            + 2 * z[0] ** 2  # penalty on acc_left
            + 2 * z[1] ** 2)  # penalty on acc_right


def set_bounds():
    '''z = [acc,v_delta,xpos, ypos, theta, v, delta]'''
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


def create_model():
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel()
    model.N = 15  # horizon length
    model.nvar = 7  # number of variables
    model.neq = 5  # number of equality constraints
    model.npar = 2  # number of runtime parameters

    # Objective function
    model.objective = obj
    model.objectiveN = objN  # increased costs for the last stage

    # We use an explicit RK4 integrator here to discretize continuous dynamics
    model.continuous_dynamics = continuous_dynamics
    
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((5, 2)), np.eye(5)], axis=1)

    # Inequality constraints
    #  upper/lower variable bounds lb <= z <= ub
    all_bounds = set_bounds()
    model.lb = all_bounds[:, 0]
    model.ub = all_bounds[:, 1]

    # Initial condition on vehicle states x
    model.xinitidx = range(2, 7)  # use this to specify on which variables initial conditions
    # are imposed
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
    codeoptions.maxit = 200  # Maximum number of iterations
    codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but
    #                             not for timings)
    codeoptions.optlevel = 0  # 0 no optimization, 1 optimize for size,
    #                             2 optimize for speed, 3 optimize for size & speed
    codeoptions.cleanup = True
    codeoptions.timing = 1
    codeoptions.nlp.hessian_approximation = 'bfgs'
    codeoptions.solvemethod = 'SQP_NLP'  # choose the solver method Sequential
    #                              Quadratic Programming
    codeoptions.nlp.bfgs_init = 1 * np.identity(8)
    codeoptions.sqp_nlp.maxqps = 1  # maximum number of quadratic problems to be solved
    codeoptions.sqp_nlp.reg_hessian = 5e-9  # increase this if exitflag=-8

    codeoptions.nlp.integrator.type = 'ERK4'
    codeoptions.nlp.integrator.Ts = 0.1
    codeoptions.nlp.integrator.nodes = 10
    # change this to your server or leave uncommented for using the
    # standard embotech server at https://forces.embotech.com
    # codeoptions.server = 'https://forces.embotech.com'

    # Creates code for symbolic model formulation given above, then contacts
    # server to generate new solver
    solver = model.generate_solver(options=codeoptions)

    return model, solver
