import json
from copy import deepcopy
from datetime import datetime
from typing import List

from dg_commons.sim.simulator import SimContext
from trajectory_games import get_metrics_set
from trajectory_games.scenarios import *

# from trajectory_games.scenarios import get_scenario_4_way_crossing_uncertain_NE


def evaluate_context(result: SimContext, ego_belief, method: str):
    evaluation = {}

    evaluation["collision"] = len(result.collision_reports) > 0
    belief_dict_json = {}
    for key, val in ego_belief.p.items():
        belief_dict_json[key.pref_str] = float(val)
    evaluation["ego belief"] = belief_dict_json
    evaluation["method"] = method
    return evaluation


def test_scenario_4_way_crossing_game_playing_agent():
    """
    4 way crossing with a stop or go agent a game playing agent
    """
    sim_context = get_scenario_4_way_crossing_game_playing_agent()
    run_scenario_without_compmake(sim_context)


# def test_scenario_4_way_crossing_stochastic_game_playing_agent():
#     """
#     4 way crossing with a stop or go agent and a game playing agent that solved multiple games and creates equilibria
#     distributions depending on belief about type of other
#     """
#     sim_context = get_scenario_4_way_crossing_uncertain_NE()
#     run_scenario_without_compmake(sim_context)


# def test_scenario_4_way_crossing_uncertain_outcome_agent():
#     """
#     4 way crossing with stop or go agent, and agent making decision on uncertain outcomes
#     """
#     sim_context = get_scenario_4_way_crossing_uncertain_outcome_agent()
#     now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
#     output_dir = "experiments/" + now_str
#     run_scenario_without_compmake(sim_context, output_dir=output_dir)


def test_scenario_4_way_crossing_uncertain_outcome_agent():
    """
    4 way crossing with trajectory following agent, and agent making decision on uncertain outcomes
    """
    sim_context = get_scenario_4_way_crossing_uncertain_outcome_agent()
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    output_dir = "experiments/" + now_str
    run_scenario_without_compmake(sim_context, output_dir=output_dir)


def test_4_way_crossing_uncertain_outcome_agent_campaign():
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    output_dir = "experiments/campaigns/" + now_str

    prob_go_belief = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    action_selection_method = ["avg", "argmax"]

    result_dicts = []

    for p_go_belief in prob_go_belief:
        belief_distr = get_ego_belief_distr(p_go_belief)
        for method in action_selection_method:
            sim_context = get_scenario_4_way_crossing_uncertain_outcome_agent(
                selection_method=method, belief_distr=belief_distr
            )
            sim_context_result = run_scenario_and_return_context(sim_context, output_dir=output_dir, write_report=False)
            result_dicts.append(evaluate_context(sim_context_result, ego_belief=belief_distr, method=method))

    filename = output_dir + ".json"
    with open(filename, "w") as write_file:
        for res in result_dicts:
            json.dump(res, write_file, indent=4)


def test_4_way_crossing_uncertain_outcome_agent_campaign_2():
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    output_dir = "experiments/campaigns/" + now_str

    prob_go_belief = [0, 0.33, 0.5, 0.66, 1.0]
    action_selection_method = ["avg", "argmax"]

    result_dicts = []

    for p_go_belief in prob_go_belief:
        belief_distr = get_ego_belief_distr(p_go_belief)
        for method in action_selection_method:
            sim_context = get_scenario_4_way_crossing_uncertain_outcome_agent(
                selection_method=method, belief_distr=belief_distr
            )
            sim_context_result = run_scenario_and_return_context(sim_context, output_dir=output_dir, write_report=True)
            result_dicts.append(evaluate_context(sim_context_result, ego_belief=belief_distr, method=method))

    filename = output_dir + ".json"
    with open(filename, "w") as write_file:
        for res in result_dicts:
            json.dump(res, write_file, indent=4)


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

    return get_game_statistics(simulation_results)


if __name__ == "__main__":
    # test_scenario_4_way_crossing_uncertain_outcome_agent()
    # test_4_way_crossing_uncertain_outcome_agent_campaign()
    # test_4_way_crossing_uncertain_outcome_agent_campaign_2()
    # test_scenario_4_way_crossing_stochastic_game_playing_agent() #leave commented
    test_scenario_4_way_crossing_game_playing_agent()
    # simulation_results = test_scenario_multiple_type_beliefs()
