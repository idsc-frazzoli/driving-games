from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from crash.scenarios import get_scenario_suicidal_pedestrian, get_scenario_illegal_turn, get_scenario_bicycle
from sim.simulator import SimContext


@dataclass
class CrashingExperiment:
    name: str
    sub_experiments: Dict[str, SimContext]


def get_exp_suicidal_pedestrian() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_suicidal_pedestrian()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="suicidal-pedestrian",
        sub_experiments=sub_experiments
    )


def get_exp_illegal_turn() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_illegal_turn()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="illegal-turn",
        sub_experiments=sub_experiments
    )


def get_exp_bicycle() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_bicycle()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    # todo variation with baseline2 agent instead of ego
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(
        name="scenario-bicycle",
        sub_experiments=sub_experiments
    )
