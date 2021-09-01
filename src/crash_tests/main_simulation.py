from crash.experiments import run_scenario
from crash.scenarios import *
from sim import SimParameters, SimTime
from sim.scenarios.factory import get_scenario_commonroad_replica


def commonroad_replica():
    sim_param = SimParameters(dt=SimTime("0.01"),
                              dt_commands=SimTime("0.05"),
                              max_sim_time=SimTime(6),
                              sim_time_after_collision=SimTime(6))
    # initialize all contexts/ agents and simulator
    sim_context = get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml", sim_param=sim_param)
    run_scenario(sim_context)


def az_optimal_crashing():
    sim_context = get_scenario_az_01()
    run_scenario(sim_context)


def em_devel():
    sim_context = get_scenario_03()
    run_scenario(sim_context)


if __name__ == '__main__':
    #run_experiments_with_compmake()
    az_optimal_crashing()
