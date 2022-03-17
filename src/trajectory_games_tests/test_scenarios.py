from crash.experiments import run_scenario_without_compmake


def test_scenario_two_lanes():
    sim_context = get_scenario_4_way_crossing_stochastic()
    run_scenario_without_compmake(sim_context)




if __name__ == "__main__":
    test_scenario_4_way_crossing_stochastic()
