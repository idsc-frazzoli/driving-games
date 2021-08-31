import os
from datetime import datetime

from compmake import Context

from crash.reports import generate_report
from crash.scenarios import get_scenario_az_01, get_scenario_03
from sim.log_visualisation import create_vectors_log, get_input_plots, get_state_plots
from sim.simulator import SimContext, Simulator


def run_scenario(sim_context: SimContext, output_dir: str = "out"):
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    players_states_log, players_actions_log = create_vectors_log(sim_context.log, sim_context.players)

    inputs = get_input_plots(players_actions_log)
    inputs.show()

    states = get_state_plots(players_states_log)
    states.show()


    # generate collisions and damages report
    report = generate_report(sim_context)

    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)


def run_experiments_with_compmake():
    c = Context()
    scenarios = [get_scenario_az_01(), get_scenario_03()]

    for scenario in scenarios:
        c.comp(run_scenario, scenario)
    c.compmake_console()
