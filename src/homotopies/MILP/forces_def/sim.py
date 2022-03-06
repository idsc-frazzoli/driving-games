import numpy as np
from homotopies.MILP.forces_def.sim_utils import *
from homotopies.MILP.forces_def.inequalities import get_ineq
from homotopies.MILP.forces_def.parameters import params
from homotopies.MILP.forces_def.forces_utils import ForcesException
import importlib


def sim(module_name, n_controlled, n_inter, trajs, intersects, x0, h, use_bin_init=True, use_homo=True):
    solver_module = importlib.import_module(module_name)
    problem = solver_module.test_params

    sim_time = 80

    X = np.zeros((params.n_states * n_controlled, sim_time + 1))
    X_plans = np.zeros((params.n_states * n_controlled, params.N + 1, sim_time))
    dds_plans = np.zeros((params.n_cinputs * n_controlled, params.N, sim_time))
    bin_plans = np.zeros((params.n_binputs * n_inter, params.N, sim_time))
    solvetime = np.zeros(sim_time)
    performance = {}
    X[:, 0] = x0

    box_buffer = 1.5
    A, b = get_ineq(n_controlled, n_inter, trajs, intersects, h, box_buffer=box_buffer, use_homo=use_homo)

    # simulation
    for k in range(sim_time):
        X_plans[:, 0, k] = X[:, k]
        bin_init = get_init_bin(X[:, k], trajs, intersects, n_inter, box_buffer)
        # check if all intersections are passed
        task_status = check_status(bin_init, X[:, 0], X[:, k], trajs.keys())
        performance = update_evaluation(task_status, performance, k, dds_plans)
        if all(status for status in task_status.values()):  # if all tasks are finished, stop early
            X_plans = X_plans[:, :, :k]
            dds_plans = dds_plans[:, :, :k]
            bin_plans = bin_plans[:, :, :k]
            solvetime = solvetime[:k]
            break

        #  provide initial values
        if use_bin_init:
            problem['minus_x0'] = -np.append(bin_init, X[:, k])
        else:
            problem['minus_x0'] = -X[:, k]

        for j in range(0, params.N):
            problem['ineq_A{:02d}'.format(j + 1)] = A
            problem['ineq_b{:02d}'.format(j + 1)] = b

        [solverout, exitflag, info] = solver_module.test_solve(problem)

        if (exitflag == 1):
            X_plan, dds_plan, bin_plan = extract_plans(solverout, n_controlled, n_inter)
            X_plans[:, 1:, k] = X_plan
            dds_plans[:, :, k] = dds_plan
            bin_plans[:, :, k] = bin_plan
            solvetime[k] = info.solvetime * 1000
            print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
        else:
            print('exitflag=', exitflag)
            X_plans = X_plans[:, :, :k]
            dds_plans = dds_plans[:, :, :k]
            bin_plans = bin_plans[:, :, :k]
            solvetime = solvetime[:k]
            break
            # raise ForcesException(exitflag)

        X[:, k + 1] = update_sim(X[:, k], dds_plan[:, 0], n_controlled)
    return X_plans, dds_plans, bin_plans, solvetime, performance
