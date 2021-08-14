import os
from datetime import datetime

from crash.reports import generete_report

from crash.scenarios import *
from crash.scenarios import get_scenario_03, get_scenario_04
from sim.simulator import Simulator

if __name__ == '__main__':
    sim = Simulator()
    # initialize all contexts/ agents and simulator
    sim_context = get_scenario_03()
    # run simulations
    sim.run(sim_context)
    # generate report
    report = generete_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    output_dir = "out"
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)
