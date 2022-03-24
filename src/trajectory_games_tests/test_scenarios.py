from trajectory_games.scenarios import *
from crash.experiments import run_scenario_without_compmake


# this function generates pickle file with trajectories of states and commands for a stop and go agent
def generate_pickle_for_stop_go_agent():
    behaviors = ["stop", "go"]
    for behavior in behaviors:
        sim_context = four_way_crossing_stop_go_scenario(behavior=behavior)
        run_scenario_without_compmake(sim_context, write_report=True)


def test_scenario_4_way_crossing_stochastic():
    sim_context = get_scenario_4_way_crossing_stochastic()
    run_scenario_without_compmake(sim_context)


def test_scenario_multiple_type_beliefs():
    # sim_context = ...
    # run_scenario_without_compmake(sim_context)
    pass


if __name__ == "__main__":
    # generate_pickle_for_stop_go_agent()
    test_scenario_4_way_crossing_stochastic()
    # test_scenario_multiple_type_beliefs()
