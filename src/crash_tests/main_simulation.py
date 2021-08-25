import os
from datetime import datetime

from crash.reports import generete_report
from crash.scenarios import get_scenario_az_01
from sim import SimParameters, SimTime
from sim.scenarios.factory import get_scenario_commonroad_replica
from sim.simulator import Simulator, SimContext


def _simulate_scenario(sim_context: SimContext):
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    # generate report
    report = generete_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    output_dir = "out"
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)


def commonroad_replica():
    sim_param = SimParameters(dt=SimTime(0.01),
                              dt_commands=SimTime(0.05),
                              max_sim_time=SimTime(6),
                              sim_time_after_collision=SimTime(6))
    # initialize all contexts/ agents and simulator
    sim_context = get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml", sim_param=sim_param)
    _simulate_scenario(sim_context)


def az_optimal_crashing():
    sim_context = get_scenario_az_01()
    _simulate_scenario(sim_context)


if __name__ == '__main__':
    az_optimal_crashing()
