import get_userid
import numpy as np
from .generate_model import generate_forces_model
from .parameters import params, x_idx, ub_idx, uc_idx


def generate_solver(n_player, n_controlled, n_inter, use_bin_init = False):
    stages = generate_forces_model(n_player, n_controlled, n_inter, use_bin_init)
    # Solver generation
    # -----------------
    # set output
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            z_idx = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx
            stages.newOutput('u{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx+1),
                             [stage_idx + 1],
                             [z_idx + uc_idx.ddS + 1])
            stages.newOutput('x{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx+1),
                             [stage_idx + 1],
                             [z_idx + x_idx.S + 1, z_idx + x_idx.dS + 1])
        for i_idx in range(n_inter):
            z_idx = (params.n_binputs + params.n_slacks) * i_idx
            stages.newOutput('bin{stage:02d}{inter:01d}'.format(stage=stage_idx + 1, inter=i_idx + 1),
                             [stage_idx + 1],
                             list(range(z_idx + 1, z_idx + params.n_binputs + 1)))
    # solver settings
    stages.codeoptions['name'] = 'test'
    stages.codeoptions['overwrite'] = 1
    stages.codeoptions['printlevel'] = 0
    # generate code
    stages.generateCode(get_userid.userid)


def extract_plans(solverout, n_controlled, n_inter):
    X_plan = np.zeros((params.n_states * n_controlled, params.N))
    dds_plan = np.zeros((params.n_cinputs * n_controlled, params.N))
    bin_plan = np.zeros((params.n_binputs * n_inter, params.N))
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            X_plan[params.n_states * p_idx:params.n_states * (p_idx + 1), stage_idx] = \
                solverout['x{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1)]
            dds_plan[params.n_cinputs * p_idx:params.n_cinputs * (p_idx + 1), stage_idx] = \
                solverout['u{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1)]
        for i_idx in range(n_inter):
            bin_plan[params.n_binputs * i_idx:params.n_binputs * (i_idx+1), stage_idx] = \
                solverout['bin{stage:02d}{inter:01d}'.format(stage=stage_idx + 1, inter=i_idx + 1)]

    return X_plan, dds_plan, bin_plan
