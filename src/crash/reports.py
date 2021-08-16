import matplotlib.pyplot as plt
from reprep import Report, MIME_PNG, MIME_GIF

from crash.metrics import malliaris_one
from sim.collision_visualisation import plot_collision
from sim.simulator import SimContext
from sim.simulator_animation import create_animation


def generate_metrics_report(sim_context: SimContext):
    """
    Generate One MetricsReport for each existing CollisionReport
    and save all of them as a list in sim_context.metrics_reports
    """
    for report in sim_context.collision_reports:
        colliding_players = []
        for key, value in report.players:
            colliding_players.append(key+value)
        a_state = sim_context.log.at(report.at_time)[colliding_players[0]].state
        b_state = sim_context.log.at(report.at_time)[colliding_players[1]].state
        metrics_report = malliaris_one(colliding_players[0], colliding_players[1], report, a_state, b_state)
        sim_context.metrics_reports.append(metrics_report)


def generete_report(sim_context: SimContext) -> Report:
    r = Report("Optimal crashing")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")

    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Simulation", MIME_GIF) as fn:
        create_animation(file_path=fn,
                         sim_context=sim_context,
                         figsize=(16, 8),
                         dt=20,
                         dpi=120,
                         plot_limits=None)
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
