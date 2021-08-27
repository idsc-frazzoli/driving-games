import matplotlib.pyplot as plt
import networkx as nx
from reprep import Report, MIME_PNG, MIME_GIF

from crash import logger
from crash.collisions_investigation import investigate_collision_report
from crash.metrics_malliaris import compute_malliaris_one
from crash.metrics_structures import CollMetricsReport
from sim import SimulationLog, CollisionReport
from sim.collision_visualisation import plot_collision
from sim.simulator import SimContext
from sim.simulator_animation import create_animation


def compute_damage_metrics(coll_report: CollisionReport, sim_log: SimulationLog):
    """
    Generate One MetricsReport for each existing CollisionReport
    and save all of them as a list in sim_context.metrics_reports
    """
    a, b = coll_report.players.keys()
    a_state = sim_log.at(coll_report.at_time)[a].state
    b_state = sim_log.at(coll_report.at_time)[b].state
    states = {a: a_state, b: b_state}
    # Malliaris
    malliaris = compute_malliaris_one(coll_report, states)

    return CollMetricsReport(malliaris=malliaris)


def generate_report(sim_context: SimContext) -> Report:
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
                         plot_limits="auto")
    r.add_child(get_collision_reports(sim_context))
    return r


def get_collision_reports(sim_context: SimContext) -> Report:
    r = Report("Collision report")
    logger.info(f"Generating collision images for {len(sim_context.collision_reports)} collisions")
    accidents_report, coll_graph = investigate_collision_report(sim_context.collision_reports, True)

    with r.plot(nid="CollisionGraph", mime=MIME_PNG) as pylab:
        node_colors = ["darkred" if _["at_fault"] else "forestgreen" for _ in coll_graph.nodes.values()]
        edgecolors = [sim_context.models[_].get_geometry().color for _ in coll_graph.nodes]
        nx.draw(coll_graph, with_labels=True, node_color=node_colors, edgecolors=edgecolors)
    plt.close()

    for i, report in enumerate(accidents_report):
        r.text(f"Collision-{i}", text=report.__str__())
        with r.data_file(f"Collision-{i}-viz", MIME_PNG) as f:
            plot_collision(report)
            plt.savefig(f)
        plt.close()
        damage_metrics = compute_damage_metrics(coll_report=report, sim_log=sim_context.log)
        r.text(f"Collision-{i}-damages", text=damage_metrics.__str__())

    return r
