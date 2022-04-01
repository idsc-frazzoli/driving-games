import pickle
from copy import deepcopy
from typing import List

from crash.experiments import run_scenario_without_compmake, run_scenario_and_return_context
from dg_commons.sim.simulator import SimContext
from trajectory_games import get_metrics_set
from trajectory_games.scenarios import *
from trajectory_games.simulation_campaign import get_game_statistics


def test_scenario_4_way_crossing_stochastic():
    sim_context = get_scenario_4_way_crossing_stochastic()
    run_scenario_without_compmake(sim_context)


def test_scenario_multiple_type_beliefs():
    sim_context_set = get_scenario_4_way_crossing_stochastic_multiple_type_beliefs()
    simulation_results: List[SimContext] = []
    temp_obj_large = {}
    for idx, sim_context in enumerate(sim_context_set):
        result = run_scenario_and_return_context(sim_context, write_report=False)
        all_metrics = get_metrics_set()
        temp_obj = {}
        for metric in all_metrics:
            temp_obj[metric] = metric.cache

        temp_obj_large[idx] = deepcopy(temp_obj)
        simulation_results.append(result)

    # for debugging
    # with open('all_metrics_5_exps.pickle', 'wb') as f:
    #     pickle.dump(temp_obj_large, f)
    #     # from trajectory_games.simulation_campaign import get_game_statistics
    #     # get_game_statistics([result])

    # return 0
    return get_game_statistics(simulation_results)


if __name__ == "__main__":
    # test_scenario_4_way_crossing_stochastic()
    simulation_results = test_scenario_multiple_type_beliefs()
