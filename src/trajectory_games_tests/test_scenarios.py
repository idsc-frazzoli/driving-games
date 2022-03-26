from typing import Mapping, List

from dg_commons import PlayerName
from dg_commons.sim.simulator import SimContext
from trajectory_games.scenarios import *
from crash.experiments import run_scenario_without_compmake, run_scenario_from_campaign
from trajectory_games.simulation_campaign import *


def test_scenario_4_way_crossing_stochastic():
    sim_context = get_scenario_4_way_crossing_stochastic()
    run_scenario_without_compmake(sim_context)


def test_scenario_multiple_type_beliefs():
    EGO = PlayerName("Ego")
    P1 = PlayerName("P1")
    types_of_others: Mapping[PlayerName, List[str]] = {
        EGO: ["pref_granny", "pref_granny_level_1"],
        P1: ['pref_granny_level_1', 'pref_granny_level_2']
    }
    campaign_params: SimulationCampaignParams= SimulationCampaignParams(
        n_experiments=10,
        types_of_others=types_of_others

    )
    sim_context_set = get_simulation_campaign_from_params(campaign_params)
    simulation_results: List[SimContext] = []
    for sim_context in sim_context_set:
        result = run_scenario_from_campaign(sim_context)
        simulation_results.append(result)
    return simulation_results
    # return get_game_statistics(simulation_results)



if __name__ == "__main__":
    # generate_pickle_for_stop_go_agent()
    # test_scenario_4_way_crossing_stochastic()
    simulation_results =test_scenario_multiple_type_beliefs()
    a=10
