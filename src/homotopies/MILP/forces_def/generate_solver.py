import get_userid
from .generate_model import generate_forces_model
from .parameters import params, x_idx, ub_idx, uc_idx


def generate_solver(n_player, n_controlled, n_inter, use_bin_init=False):
    # generate model
    stages = generate_forces_model(n_player, n_controlled, n_inter, use_bin_init)

    # set output
    for stage_idx in range(params.N):
        for p_idx in range(n_controlled):
            z_idx = (params.n_binputs + params.n_slacks) * n_inter + (params.n_cinputs + params.n_states) * p_idx
            stages.newOutput('u{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1),
                             [stage_idx + 1],
                             [z_idx + uc_idx.ddS + 1])
            stages.newOutput('x{stage:02d}{player:01d}'.format(stage=stage_idx + 1, player=p_idx + 1),
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
    stages.codeoptions.mip['queuesize'] = 1000  # increase it if solver returns exitflag=-2
    # generate code
    stages.generateCode(get_userid.userid)
