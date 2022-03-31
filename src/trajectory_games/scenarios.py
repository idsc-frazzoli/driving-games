import os
from decimal import Decimal as D
from itertools import product
from math import pi
from typing import List, Mapping, Optional

import matplotlib
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt

from dg_commons import PlayerName, Timestamp
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
from trajectory_games import TrajectoryGenParams
from trajectory_games.agents.game_playing_agent import GamePlayingAgent
from trajectory_games.agents.stop_or_go_agent import StopOrGoAgent
from trajectory_games.simulation_campaign import *

__all__ = [
    "get_scenario_4_way_crossing_stochastic",
    "get_scenario_4_way_crossing_stochastic_multiple_type_beliefs",
]

from trajectory_games.structures import TrajectoryGamePosetsParam

P1, EGO = (
    PlayerName("P1"),
    PlayerName("Ego"),
)

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


def get_scenario_4_way_crossing_stochastic(pref_structures: Optional[Mapping[PlayerName, str]] = None,
                                           sim_params: Optional[SimParameters] = None,
                                           receding_horizon_time: Optional[Timestamp] = None,
                                           ) -> SimContext:
    scenario_name = "DEU_Ffb-1_7_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)

    # seed for random number generation
    # seed = 0

    # probability that agent will go and not stop
    prob_go = 0.5

    plot = False
    draw_labels = True

    x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2.0, vx=kmh2ms(20), delta=0)
    x0_ego = VehicleStateDyn(x=42.0, y=0.0, theta=0.0, vx=kmh2ms(30), delta=0)

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
    if sim_params is None:
        sim_params = SimParameters(dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(4))

    # todo: look into seed
    # random.seed(a=seed)
    # unif = random.uniform(0, 1)
    # if unif > prob_go:
    #     behavior = "stop"
    # else:
    #     behavior = "go"

    ref_lanes: Mapping[PlayerName, RefLaneGoal] = {}
    if pref_structures is None:
        pref_structures = {
            P1: "pref_granny",
            EGO: "pref_granny",
        }

    # transition generator
    # u_acc = frozenset([1.0, 2.0])
    u_acc = frozenset([1.0, 2.0])
    u_dst = frozenset([-0.5, 0.5])
    params = TrajectoryGenParams(
        solve=False,
        s_final=-1,
        max_gen=100,
        dt=D("0.8"),
        # keep at max 1 sec, increase k_maxgen in trajectory_generator for having more generations
        u_acc=u_acc,
        u_dst=u_dst,
        v_max=ego_model.vp.vx_limits[1],
        v_min=ego_model.vp.vx_limits[0],
        st_max=ego_model.vp.delta_max,
        dst_max=ego_model.vp.ddelta_max,
        dt_samp=D("0.2"),
        dst_scale=False,
        n_factor=0.8,
        vg=VehicleGeometry.default_car(),
    )

    # params = TrajectoryGenParams(
    #     solve=False,
    #     s_final=-1,
    #     max_gen=100,
    #     dt=D("1.0"),
    #     # keep at max 1 sec, increase k_maxgen in trajectory_generator for having more generations
    #     u_acc=u_acc,
    #     u_dst=u_dst,
    #     v_max=15.0,
    #     v_min=0.0,
    #     st_max=0.5,
    #     dst_max=1.0,
    #     dt_samp=D("0.2"),
    #     dst_scale=False,
    #     n_factor=0.8,
    #     vg=VehicleGeometry.default_car(),
    # )

    u_acc_2 = frozenset([1.0])
    u_dst_2 = frozenset([0.0])
    params_2 = TrajectoryGenParams(
        solve=False,
        s_final=-1,
        max_gen=100,
        dt=D("1.0"),
        # keep at max 1 sec, increase k_maxgen in trajectrory_generator for having more generations
        u_acc=u_acc_2,
        u_dst=u_dst_2,
        v_max=15.0,
        v_min=0.0,
        st_max=0.5,
        dst_max=1.0,
        dt_samp=D("0.2"),
        dst_scale=False,
        n_factor=0.8,
        vg=VehicleGeometry.default_car(),
    )
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
        P1: params,
        EGO: params,
    }
    # compute all reference lanes
    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        ref_lanes[agent] = RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=1000)

    game_params_ego = TrajectoryGamePosetsParam(
        map_name="DEU_Ffb-1_7_T-1",
        initial_states=initial_states,
        ref_lanes=ref_lanes,
        pref_structures=pref_structures,
        traj_gen_params=traj_gen_params,
        n_traj_max=5,
        refresh_time=receding_horizon_time,
        sampling_method="uniform"
    )

    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        if agent == P1:
            agents.append(StopOrGoAgent(
                ref_lane=dglane_from_position(p, net, succ_lane_selection=2),
                prob_go=prob_go,
                # behavior=behavior,
            ))
        if agent == EGO:
            agents.append(GamePlayingAgent(
                game_params=game_params_ego
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


def get_simulation_campaign_from_params(params: SimulationCampaignParams) -> List[SimContext]:
    sim_contexts = []
    player_types = params.player_types.values()
    player_names = list(params.player_types.keys())
    for combination in product(*player_types):
        type_combination: Mapping[PlayerName, str] = {player_names[i]: combination[i] for i in range(len(player_names))}
        sim_contexts.append(
            get_scenario_4_way_crossing_stochastic(
                pref_structures=type_combination,
                sim_params=params.sim_params,
                receding_horizon_time=params.receding_horizon_time
            )
        )

    return sim_contexts


def get_scenario_4_way_crossing_stochastic_multiple_type_beliefs():
    EGO = PlayerName("Ego")
    P1 = PlayerName("P1")
    player_types: Mapping[PlayerName, List[str]] = {
        EGO: ["pref_leon_dev_4"],
        P1: ["pref_leon_dev", "pref_leon_dev_1", "pref_leon_dev_2", "pref_leon_dev_3", "pref_leon_dev_4"]
    }

    campaign_params: SimulationCampaignParams = SimulationCampaignParams(
        n_experiments=10, # for now not used -> use for statistics
        player_types=player_types

    )
    sim_context_set = get_simulation_campaign_from_params(campaign_params)
    return sim_context_set
