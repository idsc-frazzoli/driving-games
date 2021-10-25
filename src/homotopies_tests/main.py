import os
from datetime import datetime

from dg_commons.sim.simulator import Simulator
from homotopies.report import generate_report
from homotopies.scenarios import get_homotopy_scenario

if __name__ == "__main__":
    output_dir = "out"
    sim_context = get_homotopy_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)
