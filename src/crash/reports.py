from reprep import Report, MIME_GIF

from sim.simulator import SimContext
from sim.simulator_animation import create_animation


def generete_report(sim_context: SimContext) -> Report:
    r = Report("Optimal crashing")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")

    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Simulation", MIME_GIF) as fn:
        create_animation(file_path=fn,
                         sim_context=sim_context,
                         fig_size=None,
                         dt=20,
                         dpi=120)
    return r
