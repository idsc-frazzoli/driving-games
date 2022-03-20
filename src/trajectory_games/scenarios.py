from math import pi
import os
from decimal import Decimal as D
import random
from typing import List, Mapping, Optional

import matplotlib
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt

from crash.agents import B2Agent
from dg_commons import PlayerName
from dg_commons.controllers.speed import SpeedController, SpeedControllerParam
from dg_commons.controllers.steer import SteerController, SteerControllerParam
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.agents import Agent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import dglane_from_position
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_structures import SimParameters

from dg_commons_dev.utils import get_project_root_dir
from trajectory_games.agents.game_playing_agent import GamePlayingAgent
from trajectory_games.agents.stop_or_go_agent import StopOrGoAgent

__all__ = [
    "get_scenario_4_way_crossing_stochastic",
    "four_way_crossing_stop_go_scenario",
]

P1, EGO = (
    PlayerName("P1"),
    PlayerName("Ego"),
)

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


def four_way_crossing_stop_go_scenario(behavior: str) -> SimContext:
    scenario_name = "DEU_Ffb-1_7_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)

    assert behavior in ["stop", "go"], "Behavior can only be stop or go"

    x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2, vx=kmh2ms(30), delta=0)
    p1_model = VehicleModelDyn.default_car(x0=x0_p1)

    models = {P1: p1_model}

    net = scenario.lanelet_network

    agents: List[Agent] = []

    sim_params = SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10))
    stopping_time = D(6)  # at what time to get to a complete halt, depends on scenario, d(5) for: "DEU_Ffb-1_7_T-1"

    p = np.array([x0_p1.x, x0_p1.y])
    dglane = dglane_from_position(p, net, succ_lane_selection=2)
    sp_controller_param: SpeedControllerParam = SpeedControllerParam(
        setpoint_minmax=p1_model.vp.vx_limits,
        output_minmax=p1_model.vp.acc_limits,
    )

    agents.append(StopOrGoAgent(
        ref_lane=dglane,
        my_name=P1,
        max_sim_time=sim_params.max_sim_time,
        stopping_time=stopping_time,
        generative=True,
        behavior=behavior
    ))

    players = {
        P1: agents[0],
    }

    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        param=sim_params,
    )


def get_scenario_4_way_crossing_stochastic() -> SimContext:
    scenario_name = "DEU_Ffb-1_7_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)

    # seed for random number generation
    seed = 0

    # probability that agent will go and not stop
    prob_go = 0.5

    plot = True
    draw_labels = True

    x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2, vx=kmh2ms(30), delta=0)
    x0_ego = VehicleStateDyn(x=42.0, y=0.0, theta=0.00, vx=kmh2ms(30), delta=0)

    p1_model = VehicleModelDyn.default_car(x0=x0_p1)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    models = {
        P1: p1_model,
        EGO: ego_model,
    }

    net = scenario.lanelet_network

    init_p1 = VehicleState(x=x0_p1.x, y=x0_p1.y, vx=x0_p1.vx, theta=x0_p1.theta, delta=x0_p1.delta)
    init_ego = VehicleState(x=x0_ego.x, y=x0_ego.y, vx=x0_ego.vx, theta=x0_ego.theta, delta=x0_ego.delta)

    initial_states = {
        P1: init_p1,
        EGO: init_ego,
    }

    if plot:
        matplotlib.use("TkAgg")
        renderer: MPRenderer = MPRenderer()
        renderer.draw_params["trajectory"]["draw_trajectory"] = False
        renderer.draw_params["dynamic_obstacle"]["draw_shape"] = False
        if draw_labels:
            renderer.draw_params["lanelet"]["show_label"] = True

        scenario.draw(renderer)
        renderer.render()
        plt.show()

    agents: List[Agent] = []

    sim_params = SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10))

    random.seed(a=seed)
    unif = random.uniform(0, 1)
    if unif > prob_go:
        behavior = "stop"
    else:
        behavior = "go"

    ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}
    pref_structures = {
        P1: "pref_granny",
        EGO: "pref_granny",
    }
    # generate to use transition generator, stop-or-go to use precomputed trajectories (TO generalize)
    transition_gen_map = {
        P1: "stop-or-go",
        EGO: "stop-or-go",

    }
    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        ref_lanes[agent] = RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=10)
        if agent == P1:
            agents.append(StopOrGoAgent(
                ref_lane=dglane_from_position(p, net, succ_lane_selection=2),
                seed=seed,
                prob_go=prob_go,
                file_path="trajectory_stop_go.pickle",
                generative=False,
                behavior=behavior
            ))
        if agent == EGO:
            agents.append(GamePlayingAgent(
                map_name="DEU_Ffb-1_7_T-1",
                initial_states=initial_states,
                ref_lanes=ref_lanes,
                pref_structures=pref_structures,
                transition_gen_map=transition_gen_map
            ))

    players = {
        P1: agents[0],
        EGO: agents[1],
    }

    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        param=sim_params,
    )
