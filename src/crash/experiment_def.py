from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from crash.agents import B2Agent, MilleniumFalcon, B1Agent
from crash.scenarios import get_scenario_suicidal_pedestrian, get_scenario_illegal_turn, EGO, get_scenario_two_lanes
from sim.simulator import SimContext


@dataclass
class CrashingExperiment:
    name: str
    sub_experiments: Dict[str, SimContext]


def get_exp_suicidal_pedestrian() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_suicidal_pedestrian()
    # todo properly initialize B1
    baseline1_agent = B1Agent()
    sim_context.players[EGO] = baseline1_agent
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo properly initialize B2
    baseline2_agent = B2Agent()
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo properly initialize MilleniumFalcon
    mf_agent = MilleniumFalcon()
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="suicidal-pedestrian",
        sub_experiments=sub_experiments
    )


def get_exp_illegal_turn() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_illegal_turn()
    # todo properly initialize B1
    baseline1_agent = B1Agent()
    sim_context.players[EGO] = baseline1_agent
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo properly initialize B2
    baseline2_agent = B2Agent()
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo properly initialize MilleniumFalcon
    mf_agent = MilleniumFalcon()
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="illegal-turn",
        sub_experiments=sub_experiments
    )


def get_exp_two_lanes_scenario() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_two_lanes()
    # todo properly initialize B1
    #baseline1_agent = B1Agent()
    #sim_context.players[EGO] = baseline1_agent
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo properly initialize B2
    baseline2_agent = B2Agent()
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo properly initialize MilleniumFalcon
    mf_agent = MilleniumFalcon()
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="two-lanes",
        sub_experiments=sub_experiments
    )
