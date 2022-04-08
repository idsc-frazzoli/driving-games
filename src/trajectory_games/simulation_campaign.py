import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal as D
from typing import List, Mapping, Set, Optional, Any

import matplotlib
from matplotlib import pyplot as plt
from networkx import DiGraph, draw_networkx_edges, draw_networkx_labels

from dg_commons import PlayerName, Timestamp
from dg_commons.planning import Trajectory
from dg_commons.sim import SimParameters
from dg_commons.sim.simulator import SimContext
from dg_commons_dev.utils import get_project_root_dir
from driving_games.metrics_structures import EvaluatedMetric
from trajectory_games import PosetalPreference, MetricNodePreference
from trajectory_games.metrics import get_metrics_set

__all__ = [
    "SimulationCampaignParams",
]

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


@dataclass
class SimulationCampaignParams:
    """
    Parameters for running a campaign of simulations with varying parameters.
    :param n_experiments: number of time an experiment is repeated
    :param player_types: possible types for each player. All combinations (euclidean product) will be tested
    :param sim_params: parameters for the single simulation, such as time, refresh rate of commands, etc.
    :param receding_horizon_time: time interval for solving game again (receding horizon control)
    """

    n_experiments: int  # todo: use this when seed will have been implemented for all randomized steps of simulation
    player_types: Mapping[PlayerName, List[str]]
    sim_params: SimParameters = SimParameters(
        dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(5)
    )
    receding_horizon_time: Optional[Timestamp] = None


@dataclass
class SimulationCampaignStatistics:
    """
    Statistics to store results of a simulation campaign
    :param n_experiments: number of time each experiment was repeated
    :param metrics: evaluated metrics for each experiment
    """

    n_experiments: int  # todo: assign correctly when seed will be used for multiple experiments
    metrics: Mapping[int, Mapping[PlayerName, Any]]


# TODO: this needs to be tested. Separate the function summing frozen dictionaries
# TODO: fix comment
"""
Sums metrics over all games (i.e. every sampling time, when it's returned by on_get_extra in simulation)
This is to account for the receding horizon game!!

Then for each simulated scenario, returns a set of metrics for all players.
"""


def get_game_statistics(sim_results: List[SimContext]):
    EGO = PlayerName("Ego")
    campaign_metrics: Mapping[int, Mapping[PlayerName, Set[EvaluatedMetric]]] = {n: {} for n in range(len(sim_results))}
    all_metrics = get_metrics_set()

    # todo: make it work also for receding horizon. Since trajectories should be unique, change should net be great.
    for idx, result in enumerate(sim_results):
        trajs = []
        pref_str_dict = result.players[EGO].game_params.pref_structures
        for pname, pref_str in pref_str_dict.items():
            pref = PosetalPreference(pref_str, use_cache=False)
            trajectory = result.players[EGO].selected_eq.actions[pname]
            filename = "Test_" + str(idx) + "_" + str(pname) + datetime.now().strftime("%y-%m-%d-%H%M%S")
            plot(
                pref_graph=pref.graph,
                traj=trajectory,
                player_name=pname,
                show_plot=False,
                experiment_id=idx,
                filename=filename,
            )
            player_evaluated_metrics = set()
            for metric in all_metrics:
                player_evaluated_metrics.add(metric.cache[trajectory])
            if True:
                trajs.append(trajectory)
                campaign_metrics[idx][pname] = deepcopy(player_evaluated_metrics)

    return SimulationCampaignStatistics(n_experiments=99999, metrics=campaign_metrics)


def plot(
    pref_graph: DiGraph,
    traj: Trajectory,
    player_name: PlayerName,
    experiment_id: int,
    show_plot: bool = False,
    filename: Optional[str] = None,
):
    matplotlib.use("TkAgg")
    fig, ax = plt.subplots()
    plot_pref(axis=ax, pref_graph=pref_graph, traj=traj, player_name=player_name, experiment_id=experiment_id)

    if filename:
        plt.savefig(filename)
    if show_plot:
        plt.show()


def plot_pref(
    axis,
    pref_graph: DiGraph,
    traj: Trajectory,
    experiment_id: int,
    player_name: PlayerName,
):
    origin = (0, 0)
    X, Y = origin
    G = pref_graph

    def pos_node(n: MetricNodePreference):
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        return x + X, y + Y

    pos = {_: (pos_node(_)) for _ in G.nodes}

    metric_values = {}
    for node in G.nodes:
        c = list(node.weights.keys())
        for ci in c:
            metric_values[node] = ci.cache[traj]

    labels = {n: str(metric_values[n]) for n in G.nodes}

    draw_networkx_edges(G, pos=pos, edgelist=G.edges(), ax=axis, arrows=True, arrowstyle="-")

    draw_networkx_labels(G, pos=pos, labels=labels, ax=axis, font_size=8, font_color="b")

    axis.text(x=X, y=Y + 10.0, s=player_name + ".  Experiment id: " + str(experiment_id), ha="center", va="center")
    axis.set_ylim(top=Y + 15.0, bottom=Y - 40.0)
    axis.set_xlim(left=X - 70.0, right=X + 70.0)
