import forcespro
from .parameters import params, ub_idx
from .equalities import get_eq
from .objective import get_obj
from .inequalities import get_bounds


def get_bin_idx(n_inter):
    """get all indexes of the binary variables"""
    bin_idx = []
    for i_idx in range(n_inter):
        bin_idx += [i_idx * (params.n_binputs + params.n_slacks) + i.value + 1 for i in ub_idx]  # 1-indexed
    return bin_idx


def generate_forces_model(n_players, n_controlled, n_inter, use_bin_init=False, use_homo=True):
    # Problem dimensions
    stages = forcespro.MultistageProblem(params.N)  # 0-indexed
    n_var = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * n_controlled
    neq, C, D, c = get_eq(n_controlled, n_inter)
    f, H = get_obj(n_controlled, n_inter)
    bounds = get_bounds(n_controlled, n_inter)
    for i in range(params.N):
        # dimensions
        stages.dims[i]["n"] = n_var  # length of stage variable zi

        # cost
        stages.cost[i]["f"] = f[i, :]
        stages.cost[i]["H"] = H[i, :, :]

        # bounds
        stages.dims[i]["l"] = n_var  # number of lower bounds
        stages.dims[i]["u"] = n_var  # number of upper bounds
        stages.ineq[i]["b"]["lbidx"] = list(range(1, n_var + 1))  # index vector for lower bounds, 1-indexed
        stages.ineq[i]["b"]["lb"] = bounds[:, 0]  # lower bounds
        stages.ineq[i]["b"]["ubidx"] = list(range(1, n_var + 1))  # index vector for upper bounds, 1-indexed
        stages.ineq[i]["b"]["ub"] = bounds[:, 1]  # upper bounds

        # equality constraints
        if i > 0:
            stages.dims[i]["r"] = neq  # number of equality constraints
        else:  # at initial stage
            if use_bin_init:
                stages.dims[i]["r"] = neq
            else:
                stages.dims[i]["r"] = params.n_states * n_controlled  # only provide initial value of states

        if i < params.N - 1:
            stages.eq[i]["C"] = C[i, :, :]
        if i > 0:
            stages.eq[i]["c"] = c[i, :, :]
            stages.eq[i]["D"] = D[i, :, :]
        if i == 0:
            if use_bin_init:
                stages.eq[i]["D"] = D[i, :, :]
            else:
                eq_start_idx = params.n_binputs * n_inter
                eq_idx = list(range(eq_start_idx, eq_start_idx + params.n_states * n_controlled))
                stages.eq[i]["D"] = D[i, eq_idx, :]

        # inequality constraints
        if use_homo:
            stages.dims[i]["p"] = params.n_ineq * n_inter  # number of polytopic constraints
        else:
            stages.dims[i]["p"] = (params.n_ineq - 1) * n_inter
        stages.newParam("ineq_A{:02d}".format(i + 1), [i + 1], "ineq.p.A")  # set as runtime parameters
        stages.newParam("ineq_b{:02d}".format(i + 1), [i + 1], "ineq.p.b")

        # declare binary variables
        stages.bidx[i] = get_bin_idx(n_inter)  # which indices are binary? 1-indexed

    stages.newParam("minus_x0", [1], "eq.c")  # RHS of first eq. constr. is a parameter: -x0

    return stages
