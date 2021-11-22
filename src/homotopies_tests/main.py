import os
from datetime import datetime

from dg_commons.sim.simulator import Simulator
from homotopies.report import generate_report
from homotopies.scenarios import *

if __name__ == "__main__":
    output_dir = "out"
    #sim_context = get_homotopy_scenario()
    sim_context = get_intersection_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    report_file = os.path.join(output_dir, f"homotopy_{now_str}.html")
    report.to_html(report_file)
