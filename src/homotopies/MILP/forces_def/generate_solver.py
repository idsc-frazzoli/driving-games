import get_userid

from .generate_model import generate_forces_model
from .parameters import params


def generatee_solver(n_player, n_controlled, n_inter):
    stages = generate_forces_model(n_player, n_controlled, n_inter)
    # Solver generation
    # -----------------
    # set output
    stages.newOutput('u', list(range(1, params.Nstages + 1)), [uc_idx.ddS + 1])
    stages.newOutput('x', list(range(1, params.Nstages + 1)), [x_idx.S + 1, x_idx.dS + 1])
    stages.newOutput('bin', list(range(1, params.Nstages + 1)), range(ub_idx.Sigma_L + 1, ub_idx.Sigma_A + 2))
    # Set solver options
    # solver settings
    stages.codeoptions['name'] = 'MILP_MPC'
    stages.codeoptions['overwrite'] = 1
    stages.codeoptions['printlevel'] = 0
    # generate code
    stages.generateCode(get_userid.userid)