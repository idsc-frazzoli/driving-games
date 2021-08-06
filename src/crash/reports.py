import matplotlib.pyplot as plt
from reprep import Report, MIME_PNG, MIME_GIF

from sim.collision_visualisation import plot_collision
from sim.simulator import SimContext
from sim.simulator_animation import create_animation


def generete_report(sim_context: SimContext) -> Report:
    r = Report("Optimal crashing")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")

    # gif_viz = r.figure(cols=1)
    # with gif_viz.data_file("Simulation", MIME_GIF) as fn:
    #     create_animation(file_path=fn,
    #                      sim_context=sim_context,
    #                      fig_size=None,
    #                      dt=20,
    #                      dpi=120)
    r.add_child(get_collsion_reports(sim_context))
    return r


def get_collsion_reports(sim_context: SimContext) -> Report:
    r = Report("Collison report")
    for i, report in enumerate(sim_context.collision_reports):
        r.text(f"Collision-{i}", text=report.__str__())
        with r.data_file(f"Collision-{i}-viz", MIME_PNG) as f:
            plot_collision(report)
            plt.savefig(f)

    return r
