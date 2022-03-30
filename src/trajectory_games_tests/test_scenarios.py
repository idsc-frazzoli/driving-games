from typing import List

from crash.experiments import run_scenario_without_compmake, run_scenario_and_return_context
from dg_commons.sim.simulator import SimContext
from trajectory_games import get_metrics_set
from trajectory_games.scenarios import *
from trajectory_games.simulation_campaign import get_game_statistics


def test_scenario_4_way_crossing_stochastic():
    sim_context = get_scenario_4_way_crossing_stochastic()
    run_scenario_without_compmake(sim_context)
    all_metrics = get_metrics_set() #just for debugging


def test_scenario_multiple_type_beliefs():
    sim_context_set = get_scenario_4_way_crossing_stochastic_multiple_type_beliefs()
    simulation_results: List[SimContext] = []
    for sim_context in sim_context_set:
        result = run_scenario_and_return_context(sim_context, write_report=True)
        simulation_results.append(result)
    return get_game_statistics(simulation_results)


if __name__ == "__main__":
    # generate_pickle_for_stop_go_agent()
    test_scenario_4_way_crossing_stochastic()
    simulation_results = test_scenario_multiple_type_beliefs()
