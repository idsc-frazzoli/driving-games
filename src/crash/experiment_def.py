from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from crash.agents import B2Agent, MilleniumFalcon
from crash.scenarios import (
    EGO,
    get_scenario_bicycles,
    get_scenario_illegal_turn,
    get_scenario_suicidal_pedestrian,
    get_scenario_two_lanes,
)
from dg_commons.sim.simulator import SimContext


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
    b1agent = sim_context_2.players[EGO]
    baseline2_agent = B2Agent(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    b1agent = sim_context_3.players[EGO]
    mf_agent = MilleniumFalcon(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(name="suicidal-pedestrian", sub_experiments=sub_experiments)


def get_exp_illegal_turn() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_illegal_turn()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    b1agent = sim_context_2.players[EGO]
    baseline2_agent = B2Agent(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    mf_agent = MilleniumFalcon(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(name="illegal-turn", sub_experiments=sub_experiments)


def get_exp_two_lanes_scenario() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_two_lanes()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    b1agent = sim_context_2.players[EGO]
    baseline2_agent = B2Agent(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    b1agent = sim_context_3.players[EGO]
    mf_agent = MilleniumFalcon(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(name="two-lanes", sub_experiments=sub_experiments)


def get_exp_bicycles_scenario() -> CrashingExperiment:
    sub_experiments: Dict[str, SimContext] = {}
    # baseline
    sim_context = get_scenario_bicycles()
    sub_experiments.update({"baseline": sim_context})

    # baseline 2
    sim_context_2 = deepcopy(sim_context)
    b1agent = sim_context_2.players[EGO]
    baseline2_agent = B2Agent(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_2.players[EGO] = baseline2_agent
    sub_experiments.update({"baseline-avoidance": sim_context_2})

    # our
    sim_context_3 = deepcopy(sim_context)
    b1agent = sim_context_3.players[EGO]
    mf_agent = MilleniumFalcon(
        lane=b1agent.ref_lane,
        speed_controller=b1agent.speed_controller,
        steer_controller=b1agent.steer_controller,
        speed_behavior=b1agent.speed_behavior,
        pure_pursuit=b1agent.pure_pursuit,
    )
    sim_context_3.players[EGO] = mf_agent
    sub_experiments.update({"our": sim_context_3})
    return CrashingExperiment(name="bicycles", sub_experiments=sub_experiments)
