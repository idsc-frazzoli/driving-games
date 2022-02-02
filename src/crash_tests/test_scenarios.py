from crash.experiments import run_scenario_without_compmake
from crash.scenarios import *


def test_suicidal_pedestrian():
    sim_context = get_scenario_suicidal_pedestrian()
    run_scenario_without_compmake(sim_context)


def test_illegal_turn():
    sim_context = get_scenario_illegal_turn()
    run_scenario_without_compmake(sim_context)


def test_scenario_bicycle():
    sim_context = get_scenario_bicycles()
    run_scenario_without_compmake(sim_context)


def test_scenario_two_lanes():
    sim_context = get_scenario_two_lanes()
    run_scenario_without_compmake(sim_context)


def test_scenario_racetrack():
    sim_context = get_scenario_racetrack_test()
    run_scenario_without_compmake(sim_context)

def test_predictions():
    sim_context = get_scenario_predictions()
    run_scenario_without_compmake(sim_context)

def test_predictions_4waycrossing():
    sim_context = get_scenario_4waycrossing()
    run_scenario_without_compmake(sim_context)

def test_pomdp_4waycrossing():
    sim_context = get_scenario_pomdp_4waycrossing()
    run_scenario_without_compmake(sim_context)