from reprep import Report

from sim.simulator import SimContext


def generete_report(sim_context: SimContext) -> Report:

    r = Report("Optimal crashing")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")
    return r

