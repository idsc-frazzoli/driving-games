from crash.experiments import run_scenario_without_compmake
from crash.scenarios import *
from sim import SimParameters, SimTime
from sim.scenarios.factory import get_scenario_commonroad_replica


def test_suicidal_pedestrian():
    sim_context = get_scenario_suicidal_pedestrian()
    run_scenario_without_compmake(sim_context)


def test_commonroad_replica():
    sim_param = SimParameters(dt=SimTime("0.01"),
                              dt_commands=SimTime("0.05"),
                              max_sim_time=SimTime(6),
                              sim_time_after_collision=SimTime(6))
    # initialize all contexts/ agents and simulator
    sim_context = get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml", sim_param=sim_param)
    run_scenario_without_compmake(sim_context)


def test_scenario_bicycle():
    sim_context = get_scenario_bicycle()
    run_scenario_without_compmake(sim_context)


def test_em_devel():
    sim_context = get_scenario_devel()
    run_scenario_without_compmake(sim_context)
