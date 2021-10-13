import matplotlib.pyplot as plt
import networkx as nx
from reprep import Report, MIME_PNG, MIME_GIF
from zuper_commons.text import pretty_msg

from crash import logger
from crash.collisions_investigation import investigate_collision_report
from crash.metrics_malliaris import compute_malliaris_one
from crash.metrics_nieliyang import compute_NieLiYang_risk
from crash.metrics_structures import CollMetricsReport
from dg_commons.sim import CollisionReport
from dg_commons.sim.collision_visualisation import plot_collision
from dg_commons.sim.models import PEDESTRIAN, CAR, BICYCLE
from dg_commons.sim.models.model_structures import FourWheelsTypes, TRUCK
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import create_animation


def compute_damage_metrics(coll_report: CollisionReport, sim_context: SimContext) -> CollMetricsReport:
    """
    Generate One MetricsReport for each existing CollisionReport
    and save all of them as a list in sim_context.metrics_reports
    """
    sim_log = sim_context.log
    a, b = coll_report.players.keys()
    a_state = sim_log[a].states.at(coll_report.at_time)
    b_state = sim_log[b].states.at(coll_report.at_time)

    ab_modeltype = [sim_context.models[a].model_type, sim_context.models[b].model_type]
    # Check if any of the players corresponds to a Pedestrian or cyclist
    if all(m in FourWheelsTypes for m in ab_modeltype):
        malliaris = compute_malliaris_one(coll_report, states={a: a_state, b: b_state})
        nieliyang = {}
    else:
        if any([x in ab_modeltype for x in [PEDESTRIAN, BICYCLE]]) and (CAR in ab_modeltype or TRUCK in ab_modeltype):
            malliaris = {}
            nieliyang = compute_NieLiYang_risk(
                coll_report, model_types={a: sim_context.models[a].model_type, b: sim_context.models[b].model_type}
            )
        else:
            logger.info(f"We do not have a damage model for accident between {ab_modeltype}. Skipping...")
            malliaris = {}
            nieliyang = {}

    return CollMetricsReport(malliaris=malliaris, nieliyang=nieliyang)


def generate_report(sim_context: SimContext) -> Report:
    r = Report("Episode visualisation")
    if sim_context.sim_terminated is not True:
        raise RuntimeWarning("Generating a simulation report from a simulation that is not terminated")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_GIF) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 8), dt=20, dpi=120, plot_limits="auto")
    r.add_child(get_collision_reports(sim_context, skip_collision_viz=False))
    return r


def get_collision_reports(sim_context: SimContext, skip_collision_viz: bool = False) -> Report:
    """
    Generate detailed collision report
    :param sim_context:
    :param skip_collision_viz: If True speeds up skipping the visualisation of the single collision instants
    :return:
    """
    logger.info(f"Generating collision report")
    r = Report("Accidents report")
    accidents_report, coll_graph = investigate_collision_report(sim_context.collision_reports)

    fig_graph = r.figure(cols=1)
    with fig_graph.plot(nid="CollisionGraph", mime=MIME_PNG) as _:
        node_colors = ["darkred" if _["at_fault"] else "forestgreen" for _ in coll_graph.nodes.values()]
        edgecolors = [sim_context.models[_].get_geometry().color for _ in coll_graph.nodes]
        nx.draw(coll_graph, with_labels=True, node_color=node_colors, edgecolors=edgecolors)
    plt.close()

    for i, acc_report in enumerate(accidents_report):
        acc_id = "-".join(list(acc_report.players.keys()))
        r.subsection(f"Accident-{acc_id}")
        r.text(f"Accident-{acc_id}-report", text=pretty_msg(acc_report.__str__()))
        damage_metrics = compute_damage_metrics(coll_report=acc_report, sim_context=sim_context)
        r.text(f"Accident-{acc_id}-damages", text=damage_metrics.__str__())
        if not skip_collision_viz:
            collisions_wrt_accident = [
                creport for creport in sim_context.collision_reports if set(acc_report.players) == set(creport.players)
            ]
            logger.info(f"Accident {i}, generating {len(collisions_wrt_accident)} collision images")
            coll_fig = r.figure(cols=5)
            for j, coll_report in enumerate(collisions_wrt_accident):
                with coll_fig.plot(f"Collision-{i}-{j}", MIME_PNG) as _:
                    plot_collision(coll_report, sim_log=sim_context.log)
                plt.close()

    return r
