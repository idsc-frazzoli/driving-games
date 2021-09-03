import os
from datetime import datetime

from decent_params import DecentParams
from quickapp import QuickApp, QuickAppContext

from crash.experiment_def import get_exp_suicidal_pedestrian, get_exp_illegal_turn
from crash.reports import generate_report
from sim.log_visualisation import get_input_plots, get_state_plots
from sim.simulator import SimContext, Simulator

__all__ = ["CrashingExperiments", "run_crashing_experiments"]


class CrashingExperiments(QuickApp):
    """ Main Experiments runner """

    def define_options(self, params: DecentParams):
        pass

    def define_jobs_context(self, context: QuickAppContext):

        experiments = [get_exp_suicidal_pedestrian(), get_exp_illegal_turn()]
        for exp in experiments:
            cexp = context.child(exp.name, extra_report_keys=dict(experiment=exp.name))
            for scenario_name, sim_context in exp.sub_experiments.items():
                c = cexp.child(scenario_name, extra_report_keys=dict(case=scenario_name))
                sim_context = c.comp(run_simulation, sim_context)
                report = c.comp(generate_report, sim_context)
                c.add_report(report, "episode")


run_crashing_experiments = CrashingExperiments.get_sys_main()


def run_simulation(sim_context: SimContext) -> SimContext:
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    return sim_context


def run_scenario_without_compmake(sim_context: SimContext, output_dir: str = "out"):
    sim_context = run_simulation(sim_context)
    # generate collisions and damages
    inputs = get_input_plots(sim_context.log)
    inputs.show()

    states = get_state_plots(sim_context.log)
    states.show()

    report = generate_report(sim_context)
    # save report
    now_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    report_file = os.path.join(output_dir, f"optimal_crash_{now_str}.html")
    report.to_html(report_file)
