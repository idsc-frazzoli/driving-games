from reprep import Report, MIME_GIF

from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import create_animation


def generate_report(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_GIF) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 8), dt=20, dpi=120, plot_limits="auto")
    return r
