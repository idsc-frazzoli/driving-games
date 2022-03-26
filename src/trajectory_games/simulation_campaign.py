import os
from dataclasses import dataclass
from itertools import product
from typing import List, Mapping, Set, Optional, Any
from decimal import Decimal as D

from dg_commons import PlayerName, Timestamp
from dg_commons.sim import SimParameters
from dg_commons.sim.simulator import SimContext
from dg_commons_dev.utils import get_project_root_dir
from driving_games.metrics_structures import EvaluatedMetric
from trajectory_games.scenarios import get_scenario_4_way_crossing_stochastic

__all__ = [
    "SimulationCampaignParams",
    "get_simulation_campaign_from_params"

]

P1, EGO = (
    PlayerName("P1"),
    PlayerName("Ego"),
)

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


@dataclass
class SimulationCampaignParams:
    n_experiments: int  # todo: use this when seed will have been implemented for all randomized steps of simulation
    types_of_others: Mapping[PlayerName, List[str]]
    sim_params: SimParameters = SimParameters(
        dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(2)
    )
    receding_horizon_time: Optional[Timestamp] = None


@dataclass
class SimulationCampaignStatistics:
    n_experiments: int
    metrics: Mapping[int, Mapping[PlayerName, Any]] # todo define exactly


def get_simulation_campaign_from_params(params: SimulationCampaignParams) -> List[SimContext]:
    sim_contexts = []
    player_types = params.types_of_others.values()
    player_names = list(params.types_of_others.keys())
    for combination in product(*player_types):
        type_combination: Mapping[PlayerName, str] = {player_names[i]: combination[i] for i in range(len(player_names))}
        sim_contexts.append(
            get_scenario_4_way_crossing_stochastic(
                pref_structures=type_combination,
                sim_params=params.sim_params,
                receding_horizon_time=params.receding_horizon_time,
                store_metrics=True)
        )

    return sim_contexts


#TODO: this needs to be tested. Separate the function summing frozen dictionaries
"""
Sums metrics over all games (i.e. every sampling time, when it's returned by on_get_extra in simulation)
This is to account for the receding horizon game!!

Then for each simulated scenario, returns a set of metrics for all players.
"""
def get_game_statistics(sim_results: List[SimContext]):
    EGO = PlayerName("Ego")
    total_metrics: Mapping[int, Mapping[PlayerName, Any]] = {}
    for idx, result in enumerate(sim_results):
        # compute statistics only from point of view of ego (the game playing agent)
        ego_metrics = result.log[EGO].extra
        metrics: Mapping[PlayerName, Any] = {name: dict() for name in ego_metrics.values[0][0].keys()} #todo: fix type
        for element in ego_metrics.values:
            element = element[0]
            for pname in element.keys():
                if metrics == {name: dict() for name in ego_metrics.values[0][0].keys()}:
                    for pname in element.keys():
                        metrics[pname] = {k: 0.0 for k in set(element[pname])}
                        continue
                metrics[pname] = {k: metrics[pname].get(k) + element[pname].get(k).value for k in set(metrics[pname]) & set(element[pname])}
        total_metrics[idx] = metrics





    n_experiments = len(sim_results)
    statistics = SimulationCampaignStatistics(n_experiments=n_experiments, metrics=total_metrics)
    return statistics
