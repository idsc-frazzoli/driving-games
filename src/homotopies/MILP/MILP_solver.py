import forcespro
import get_userid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from enum import IntEnum
from indices import IdxState, IdxInput

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()


class MILPModelParams:
    Nstages = 15
    dT = 0.2
    x_idx: IntEnum = IdxState
    u_idx: IntEnum = IdxInput
    nx: int = len([s.value for s in IdxState])
    nu: int = len([u.value for u in IdxInput])
    nz = nx + nu
    nineq = 2 + 4  # 2 homotopic, 4 box
    neq = nx + 4  # dynamics + 4 auxiliary eqs on sigma
    H = np.zeros((nz, nz))
    H[u_idx.ddS, u_idx.ddS] = 1  # Hessian matrix in cost function
    f = np.zeros(nz)
    f[x_idx.S] = -1  # linear term in cost function
    C = np.array([[0, 1, dT],  # discretized dynamics [B,A], x_i+1 = Ax_i + Bu_i
                  [dT, 0, 1]])
    M = 1000


params = MILPModelParams
x_idx = params.x_idx
u_idx = params.u_idx


def set_bounds():
    '''
    this function returns lower and upper bounds for continuous variables [ddS, S, dS]
    '''
    acc_bounds = np.array(vehicle_params.acc_limits)
    v_bounds = np.array(vehicle_params.vx_limits)
    default_bounds = np.array([0, params.M])
    all_bounds = np.array([acc_bounds,
                           default_bounds,
                           v_bounds])
    return all_bounds


def get_ineq_A(box):
    A = np.zeros((params.nineq, params.nz))
    A[0, [u_idx.Sigma_L, u_idx.Sigma_A]] = -1
    A[1, [u_idx.Sigma_R, u_idx.Sigma_B]] = -1
    A[2, u_idx.Sigma_L] = params.M
    A[2, x_idx.S] = 1
    A[3, u_idx.Sigma_R] = box[0, 1]
    A[3, x_idx.S] = -1
    A[4, u_idx.Sigma_B] = params.M
    A[5, u_idx.Sigma_A] = box[1, 1]
    return A


def get_ineq_b(box, h, s2):
    b = np.zeros(params.nineq)
    b[0] = -1 + h
    b[1] = -h
    b[2] = params.M + box[0, 0]
    b[3] = 0
    b[4] = box[1, 0] + params.M - s2
    b[5] = s2
    return b


def create_model():
    # Model Definition
    # ----------------
    # Problem dimensions
    stages = forcespro.MultistageProblem(params.Nstages)  # 0-indexed
    for i in range(params.Nstages):
        # dimensions
        stages.dims[i]['n'] = params.nz  # length of stage variable zi
        if i > 0:
            stages.dims[i]['r'] = params.neq  # number of equality constraints
        else:
            stages.dims[i]['r'] = params.nx  # only constrain initial value of x
        stages.dims[i]['l'] = params.nz  # number of lower bounds
        stages.dims[i]['u'] = params.nz  # number of upper bounds
        stages.dims[i]['p'] = params.nineq  # number of polytopic constraints
        stages.bidx[i] = np.array([u_idx.Sigma_L + 1,
                                   u_idx.Sigma_R + 1,
                                   u_idx.Sigma_B + 1,
                                   u_idx.Sigma_A + 1])  # which indices are binary? 1-indexed

        # cost
        if i == params.Nstages - 1:
            stages.cost[i]['f'] = params.f
        else:
            stages.cost[i]['f'] = np.zeros(params.nz)
        stages.cost[i]['H'] = params.H

        # bounds
        continuous_bounds = set_bounds()
        stages.ineq[i]['b']['lbidx'] = list(range(1, params.nz + 1))  # index vector for lower bounds, 1-indexed
        stages.ineq[i]['b']['lb'] = np.concatenate((np.zeros(params.nu - 1), continuous_bounds[:, 0]))  # lower bounds
        stages.ineq[i]['b']['ubidx'] = list(range(1, params.nz + 1))  # index vector for upper bounds, 1-indexed
        stages.ineq[i]['b']['ub'] = np.concatenate((np.ones(params.nu - 1), continuous_bounds[:, 1]))  # upper bounds

        # equality constraints
        if i < params.Nstages - 1:
            C_x = np.hstack((np.zeros((params.nx, params.nu - 1)), params.C))
            C_sigma = np.hstack((-np.eye(4), np.zeros((4, params.nz - 4))))
            stages.eq[i]['C'] = np.vstack((C_x, C_sigma))
        if i > 0:
            c_x = np.zeros((params.nx, 1))
            c_sigma = np.zeros((4, 1))
            stages.eq[i]['c'] = np.vstack((c_x, c_sigma))
            D_sigma = np.hstack((np.eye(4), np.diag(np.array([1, -1, 1, -1])), np.zeros((4, 3))))
            D_x = np.hstack((np.zeros((params.nx, params.nu)), -np.eye(params.nx)))
            stages.eq[i]['D'] = np.vstack((D_x, D_sigma))
        if i == 0:
            stages.eq[i]['D'] = np.hstack((np.zeros((params.nx, params.nu)), -np.eye(params.nx)))

        # inequality constraints
        stages.newParam('ineq_A{:02d}'.format(i + 1), [i + 1], 'ineq.p.A')  # as runtime parameter
        stages.newParam('ineq_b{:02d}'.format(i + 1), [i + 1], 'ineq.p.b')

    stages.newParam('minus_x0', [1], 'eq.c')  # RHS of first eq. constr. is a parameter: -x0

    return stages


def generate_solver():
    """Generates and returns a FORCESPRO solver that calculates a path based on
    constraints and dynamics while minimizing an objective function
    """
    stages = create_model()
    # Solver generation
    # -----------------
    # set output
    stages.newOutput('u', [1], [u_idx.ddS + 1])
    stages.newOutput('x', [1], [x_idx.S + 1, x_idx.dS + 1])
    stages.newOutput('bin', [1], range(u_idx.Sigma_L + 1, u_idx.Sigma_A + 1))
    # Set solver options
    # solver settings
    stages.codeoptions['name'] = 'MILP_MPC'
    stages.codeoptions['overwrite'] = 1
    stages.codeoptions['printlevel'] = 0
    # generate code
    stages.generateCode(get_userid.userid)


if __name__ == "__main__":
    # generate_solver()
    import MILP_MPC_py

    # help(MILP_MPC_py)

    problem = MILP_MPC_py.MILP_MPC_params
    x0 = np.array([0, 0])
    kmax = 30
    X = np.zeros((params.nx, kmax + 1))
    X[:, 0] = x0
    ddS = np.zeros((1, kmax))
    bin = np.zeros((4, kmax))
    box = np.array([[1.5, 3],
                    [4, 5]])
    h = 0
    s2 = np.linspace(0, 20, params.Nstages + kmax)
    # simulation
    for k in range(0, kmax):
        problem['minus_x0'] = -X[:, k]
        for j in range(0, params.Nstages):
            problem['ineq_A{:02d}'.format(j + 1)] = get_ineq_A(box)
            problem['ineq_b{:02d}'.format(j + 1)] = get_ineq_b(box, h, s2[k + j])
        [solverout, exitflag, info] = MILP_MPC_py.MILP_MPC_solve(problem)
        if (exitflag == 1):
            ddS[:, k] = solverout['u'][0]
            print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
        else:
            print(info)
            # raise RuntimeError('Some problem in solver')

        curr = np.hstack((ddS[:, k], X[:, k])).reshape([-1, 1])
        X[:, k + 1] = np.dot(params.C, curr).reshape([2, ])

    # plot
    fig1 = plt.figure()

    plt.subplot(2, 2, 2)
    plt.step(range(0, kmax), X[1, 0:kmax], where='post')
    plt.title('states: dS')
    plt.xlim(0, kmax)
    plt.ylim(0, 36)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.axhline(y=vehicle_params.acc_limits[0], c="red", zorder=0)
    plt.axhline(y=vehicle_params.acc_limits[1], c="red", zorder=0)
    plt.step(range(0, kmax), ddS[0, 0:kmax], where='post')
    plt.title('input: ddS')
    plt.xlim(0, kmax)
    plt.ylim(1.1 * vehicle_params.acc_limits[0], 1.1 * vehicle_params.acc_limits[1])
    plt.grid()

    gs = GridSpec(2, 2, figure=fig1)
    ax = fig1.add_subplot(gs[:, 0])
    width = box[0, 1] - box[0, 0]
    height = box[1, 1] - box[1, 0]
    rect = patches.Rectangle((box[0, 0], box[1, 0]), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.plot(X[0, :kmax], s2[:kmax], 'bo-', markersize=3)
    plt.xlim([0, 15.])
    plt.ylim([0, 15.])
    plt.xlabel('s1')
    plt.ylabel('s2')

    plt.show()
