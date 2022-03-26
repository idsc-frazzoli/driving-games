import os
from dataclasses import dataclass
from itertools import product
from typing import List, Mapping, Set
from decimal import Decimal as D

from dg_commons import PlayerName
from dg_commons.sim import SimParameters
from dg_commons.sim.simulator import SimContext
from dg_commons_dev.utils import get_project_root_dir
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


@dataclass
class SimulationCampaignStatistics:
    n_experiments: int
    average_metrics: float  # todo define exactly


def get_simulation_campaign_from_params(params: SimulationCampaignParams) -> List[SimContext]:
    sim_contexts = []
    player_types = params.types_of_others.values()
    player_names = list(params.types_of_others.keys())
    for combination in product(*player_types):
        type_combination: Mapping[PlayerName, str] = {player_names[i]: combination[i] for i in range(len(player_names))}
        sim_contexts.append(
            get_scenario_4_way_crossing_stochastic(pref_structures=type_combination, sim_params=params.sim_params)
        )

    return sim_contexts


def get_game_statistics():
    statistics = SimulationCampaignStatistics()
    return statistics
