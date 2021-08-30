from crash.experiments import run_scenario
from crash.scenarios import get_scenario_az_01


def test_scenario():
    sim_context = get_scenario_az_01()
    run_scenario(sim_context, output_dir="out-tests")
