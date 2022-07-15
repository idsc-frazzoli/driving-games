import copy
import os
from decimal import Decimal as D
from fractions import Fraction
from itertools import product
from math import pi
from typing import List, Mapping, Optional

import matplotlib
import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficLightCycleElement, TrafficLightState, TrafficLight
from commonroad.visualization.mp_renderer import MPRenderer
from frozendict import frozendict
from matplotlib import pyplot as plt

from dg_commons import PlayerName, Timestamp
from dg_commons.planning import RefLaneGoal, Trajectory
from dg_commons.sim.agents import Agent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.utils import dglane_from_position
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_structures import SimParameters
from driving_games.utils import get_project_root_dir
from possibilities import ProbDist
from trajectory_games import TrajectoryGenParams, PosetalPreference, TrajectoryWorld, BicycleDynamics
from trajectory_games.agents.game_playing_agent import GamePlayingAgent
from trajectory_games.agents.stop_or_go_agent import StopOrGoAgent
from trajectory_games.agents.trajectory_following_agent import TrajectoryFollowingAgent
from trajectory_games.structures import TrajectoryGamePosetsParam

"""
This file contains functions to generate specific 4 way crossing scenarios for various experiments.
"""
__all__ = [
    "get_scenario_4_way_crossing_game_playing_agent",
    "get_scenario_4_way_crossing_stochastic_multiple_type_beliefs",
    "get_scenario_4_way_crossing_uncertain_outcome_agent",
    "get_ego_belief_distr",
    # "get_scenario_4_way_crossing_uncertain_outcome_agent_2"
]

P1, EGO, OTHER = (
    PlayerName("P1"),
    PlayerName("Ego"),
    PlayerName("Other"),
)

SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


def get_ego_belief_distr(go_belief: float):
    belief_distr = ProbDist(
        {
            PosetalPreference("go_agent", use_cache=False): Fraction(go_belief),
            PosetalPreference("stop_agent", use_cache=False): Fraction(1 - go_belief),
        }
    )

    return belief_distr


def get_curved_trajectory_horizontal(initial_state: VehicleState, params: TrajectoryGenParams):
    """
    Generate hand crafted "swerve" trajectory for 4 way crossing scenario "DEU_Ffb-1_7_T-1"
    """
    bicycle_dyn = BicycleDynamics(params=params)

    dst = 0.05
    constant_commands_turn_left = VehicleCommands(acc=0.1, ddelta=dst)
    constant_commands_turn_right = VehicleCommands(acc=0.1, ddelta=-1.5 * dst)

    dt = float(params.dt)
    max_time = dt * (params.max_gen - 1)
    time_left_steering = max_time / 3.0

    traj_and_commands = {}
    samp_traj = []
    current_state = initial_state
    values_traj = [current_state]
    timesteps_traj = [0.0]
    commands_traj = [constant_commands_turn_left]
    for time in np.arange(0, max_time, dt):
        if time < time_left_steering:
            u = constant_commands_turn_left
        else:
            u = constant_commands_turn_right
        # don't allow rear driving
        if current_state.vx < 0.0:
            u.acc = 0.0

        next_state, samp_states = bicycle_dyn.successor_ivp(
            x0=(time, current_state), u=u, dt=params.dt, dt_samp=params.dt_samp
        )
        current_state = next_state[1]
        samp_traj = samp_traj + samp_states[1:]

        for _ in range(len(samp_states[1:])):
            commands_traj.append(copy.deepcopy(u))

    for tup in samp_traj:
        values_traj.append(tup[1])
        timesteps_traj.append(tup[0])

    traj = Trajectory(values=values_traj, timestamps=timesteps_traj)
    commands = Trajectory(values=commands_traj, timestamps=timesteps_traj)
    # traj_and_commands[traj] = commands

    return traj, commands


def get_stop_or_go_trajectories_horizontal(
    initial_state: VehicleState, stopping_time: float, params: TrajectoryGenParams
):
    """
    Function to generate hand crafted trajectories for 4 way crossing scenario "DEU_Ffb-1_7_T-1"
    For the horizontal player (EGO)

    """
    bicycle_dyn = BicycleDynamics(params=params)

    assert stopping_time != 0.0

    acc_stop = -initial_state.vx / float(stopping_time)
    acc_go = 0.0
    dst = 0.0

    constant_commands = {
        "go": VehicleCommands(acc=acc_go, ddelta=dst),
        "stop": VehicleCommands(acc=acc_stop, ddelta=dst),
    }

    dt = float(params.dt)
    max_time = dt * (params.max_gen - 1)
    dict_traj = {}
    for traj_name, u in constant_commands.items():

        samp_traj = []

        current_state = initial_state
        values_traj = [current_state]
        timesteps_traj = [0.0]
        commands_traj = [u]
        for time in np.arange(0, max_time, dt):

            # don't allow rear driving
            if current_state.vx < 0.0:
                u.acc = 0.0
                current_state.vx = 0.0

            # make trajectory follow lane for 4 way crossing, for trajectory "go"
            if traj_name == "go":
                if time < 2.0:
                    u.ddelta = 0.015
                elif 2.0 < time < 4.0:
                    u.ddelta = -0.01
                else:
                    u.ddelta = -0.02

            elif traj_name == "stop":
                if current_state.vx > 0.0:
                    u.ddelta = -0.05
                else:
                    u.ddelta = 0.0

            next_state, samp_states = bicycle_dyn.successor_ivp(
                x0=(time, current_state), u=u, dt=params.dt, dt_samp=params.dt_samp
            )
            current_state = next_state[1]
            samp_traj = samp_traj + samp_states[1:]

            for _ in range(len(samp_states[1:])):
                commands_traj.append(copy.deepcopy(u))

        for tup in samp_traj:
            values_traj.append(tup[1])
            timesteps_traj.append(tup[0])

        for idx, val in enumerate(values_traj):
            if val.vx < 0.0:
                values_traj[idx].vx = 0.0

        traj = Trajectory(values=values_traj, timestamps=timesteps_traj)

        commands = Trajectory(values=commands_traj, timestamps=timesteps_traj)

        dict_traj[traj] = commands

    curved_traj, curved_command = get_curved_trajectory_horizontal(initial_state, params)
    dict_traj[curved_traj] = curved_command
    return frozendict(dict_traj)


def get_stop_or_go_trajectories_vertical(
    initial_state: VehicleState, stopping_time: float, params: TrajectoryGenParams
):
    """
    Function to generate hand crafted trajectories for 4 way crossing scenario "DEU_Ffb-1_7_T-1"
    For the vertical player (Non EGO)

    """
    bicycle_dyn = BicycleDynamics(params=params)

    assert stopping_time != 0.0

    acc_stop = -initial_state.vx / float(stopping_time)
    acc_go = 0.0
    dst = 0.0

    constant_commands = {
        "go": VehicleCommands(acc=acc_go, ddelta=dst),
        "stop": VehicleCommands(acc=acc_stop, ddelta=dst),
    }

    dt = float(params.dt)
    max_time = dt * (params.max_gen - 1)
    trajs_and_commands = {}
    for traj_name, u in constant_commands.items():
        dict_traj = {}
        samp_traj = []
        current_state = initial_state
        values_traj = [current_state]
        timesteps_traj = [0.0]
        commands_traj = [u]
        for time in np.arange(0, max_time, dt):

            # don't allow rear driving
            if current_state.vx < 0.0:
                u.acc = 0.0
                current_state.vx = 0.0

            # make trajectory follow lane for 4 way crossing, for trajectory "go"
            if traj_name == "go":
                if time < 2.0:
                    u.ddelta = -0.01
                elif 2.0 < time < 4.0:
                    u.ddelta = 0.015
                else:
                    u.ddelta = 0.015

            elif traj_name == "stop":
                u.ddelta = -0.05

            next_state, samp_states = bicycle_dyn.successor_ivp(
                x0=(time, current_state), u=u, dt=params.dt, dt_samp=params.dt_samp
            )
            current_state = next_state[1]
            samp_traj = samp_traj + samp_states[1:]

            for _ in range(len(samp_states[1:])):
                commands_traj.append(copy.deepcopy(u))

        for tup in samp_traj:
            values_traj.append(tup[1])
            timesteps_traj.append(tup[0])

        traj = Trajectory(values=values_traj, timestamps=timesteps_traj)
        commands = Trajectory(values=commands_traj, timestamps=timesteps_traj)
        dict_traj[traj] = commands
        trajs_and_commands[traj_name] = frozendict(dict_traj)

    return trajs_and_commands


def add_traffic_light_custom(scenario: Scenario) -> Scenario:
    green: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.GREEN, duration=1)
    yellow: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.YELLOW, duration=1)
    red: TrafficLightCycleElement = TrafficLightCycleElement(state=TrafficLightState.RED, duration=1)
    cycle = [red, green, yellow]
    position = np.array([73.0, -8.0])
    traffic_light: TrafficLight = TrafficLight(traffic_light_id=0, cycle=cycle, position=position)
    # for i in range(10):
    #     print(traffic_light.get_state_at_time_step(i))
    scenario.add_objects(traffic_light, lanelet_ids={49570})
    return scenario


def get_scenario_4_way_crossing_game_playing_agent(
    pref_structures: Optional[Mapping[PlayerName, str]] = None,
    sim_params: Optional[SimParameters] = None,
    receding_horizon_time: Optional[Timestamp] = None,
) -> SimContext:
    """
    Generate 4 way crossing scenario "DEU_Ffb-1_7_T-1" with a GamePlayingAgent and a StopOrGoAgent
    """
    scenario_name = "DEU_Ffb-1_7_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)
    scenario = add_traffic_light_custom(scenario)

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
        sim_params = SimParameters(
            dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2), max_sim_time=D(4.5)
        )

    ref_lanes: Mapping[PlayerName, List[RefLaneGoal]] = {}
    if pref_structures is None:
        pref_structures = {
            P1: "only_squared_acc",
            EGO: "only_squared_acc",
        }

    # trajectory generator
    u_acc = frozenset([-2.0, 2.0])
    u_dst = frozenset([-0.5, 0.5])
    params = TrajectoryGenParams(
        solve=False,
        s_final=-1,
        max_gen=5,
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
        n_factor=1.0,
        vg=VehicleGeometry.default_car(),
        v_switch=4.5,
        acc_max=11.5,
    )

    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
        P1: params,
        EGO: params,
    }
    # compute all reference lanes
    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        ref_lanes[agent] = [RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=1000)]

    game_params_ego = TrajectoryGamePosetsParam(
        scenario=DgScenario(scenario),
        initial_states=initial_states,
        ref_lanes=ref_lanes,
        pref_structures=pref_structures,
        refresh_time=1.5,
        traj_gen_params=traj_gen_params,
        n_traj_max=10,
        sampling_method="uniform",
    )

    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        if agent == P1:
            agents.append(
                StopOrGoAgent(
                    ref_lane=dglane_from_position(p, net, succ_lane_selection=2),
                    prob_go=prob_go,
                    behavior="stop",
                )
            )
        if agent == EGO:
            agents.append(GamePlayingAgent(game_params=game_params_ego))

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


# def get_scenario_4_way_crossing_uncertain_outcome_agent(pref_structures: Optional[Mapping[PlayerName, str]] = None,
#                                                         sim_params: Optional[SimParameters] = None,
#                                                         receding_horizon_time: Optional[Timestamp] = None,
#                                                         prob_go: float = 0.5,
#                                                         selection_method: Optional[str] = None,
#                                                         belief_distr: Optional[ProbDist] = None,
#                                                         ) -> SimContext:
#     """
#     Generate 4 way crossing scenario "DEU_Ffb-1_7_T-1" with an UncertainOutcomeAgent and a StopOrGoAgent (old version).
#     Trajectories are generated with trajectory generator
#     """
#     scenario_name = "DEU_Ffb-1_7_T-1"
#     scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)
#
#     plot = False
#     draw_labels = True
#
#     # initial Dynamic Vehicle States
#     x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2.0, vx=kmh2ms(15), delta=0)
#     x0_ego = VehicleStateDyn(x=45.0, y=0.0, theta=0.0, vx=kmh2ms(25), delta=0)
#
#     # vehicle dynamics and geometries
#     p1_model = VehicleModelDyn.default_car(x0=x0_p1)
#     vg_ego = VehicleGeometry.default_car(color="firebrick")
#     vg_p1 = VehicleGeometry.default_car(color="firebrick")
#     ego_model = VehicleModelDyn.default_car(x0_ego)
#     ego_model.vg = vg_ego
#
#     geos = {
#         EGO: vg_ego,
#         P1: vg_p1
#     }
#
#     models = {
#         P1: p1_model,
#         EGO: ego_model
#     }
#
#     # initial Vehicle States
#     init_p1 = VehicleState(x=x0_p1.x, y=x0_p1.y, vx=x0_p1.vx, theta=x0_p1.theta, delta=x0_p1.delta)
#     init_ego = VehicleState(x=x0_ego.x, y=x0_ego.y, vx=x0_ego.vx, theta=x0_ego.theta, delta=x0_ego.delta)
#
#     initial_states = {
#         P1: init_p1,
#         EGO: init_ego
#     }
#
#     # optional plotting
#     if plot:
#         matplotlib.use("TkAgg")
#         renderer: MPRenderer = MPRenderer()
#         renderer.draw_params["trajectory"]["draw_trajectory"] = False
#         renderer.draw_params["dynamic_obstacle"]["draw_shape"] = False
#         if draw_labels:
#             renderer.draw_params["lanelet"]["show_label"] = True
#
#         scenario.draw(renderer)
#         renderer.render()
#         plt.show()
#
#     # set default simulation parameters
#     if sim_params is None:
#         sim_params = SimParameters(dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2),
#                                    max_sim_time=D(4.5))
#
#     if belief_distr is None:
#         belief_over_p1 = ProbDist({
#             PosetalPreference("go_agent", use_cache=False): Fraction(0, 1),
#             PosetalPreference("stop_agent", use_cache=False): Fraction(1, 1)
#         })
#
#         pref_distr = {P1: belief_over_p1}
#
#     else:
#         pref_distr = {P1: belief_distr}
#
#     if selection_method is None:
#         selection_method = "avg"
#
#     ego_pref = PosetalPreference("pref_intersection", use_cache=False)
#
#     # trajectory generator
#     params = TrajectoryGenParams(
#         solve=False,
#         s_final=-1,
#         max_gen=7,
#         dt=D("1.0"),
#         u_acc=frozenset([-2.0, 2.0]),
#         u_dst=frozenset([0.0]),
#         v_max=ego_model.vp.vx_limits[1],
#         v_min=0.0,
#         st_max=ego_model.vp.delta_max,
#         dst_max=ego_model.vp.ddelta_max,
#         dt_samp=D("0.2"),
#         dst_scale=False,
#         n_factor=1.0,
#         vg=VehicleGeometry.default_car(),
#         v_switch=4.5,
#         acc_max=11.5
#     )
#
#     traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
#         P1: params,
#         EGO: params,
#     }
#
#     net = scenario.lanelet_network
#     ref_lanes: Mapping[PlayerName, List[RefLaneGoal]] = {}
#     # compute all reference lanes
#     for agent in models:
#         x0 = models[agent].get_state()
#         p = np.array([x0.x, x0.y])
#         ref_lanes[agent] = [RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=1000)]
#
#     traj_world = TrajectoryWorld(map_name=scenario_name, scenario=DgScenario(scenario), geo=geos, goals=ref_lanes)
#
#     game_params_ego = TrajectoryGamePosetsParam(
#         scenario=DgScenario(scenario),
#         initial_states=initial_states,
#         ref_lanes=ref_lanes,
#         pref_structures=pref_structures,
#         traj_gen_params=traj_gen_params,
#         n_traj_max=10,
#         sampling_method="uniform"
#     )
#
#     players = {}
#     stopping_time = D(2.5)
#     for agent in models:
#         x0 = models[agent].get_state()
#         p = np.array([x0.x, x0.y])
#         if agent == P1:
#             players[P1] = StopOrGoAgent(
#                 ref_lane=dglane_from_position(p, net, succ_lane_selection=2),
#                 prob_go=prob_go,
#                 behavior="go",  # force behavior to stop/go -> probability prob_go is ignored
#                 stopping_time=stopping_time,
#                 nominal_speed=init_p1.vx  # constant speed
#             )
#
#         if agent == EGO:
#             players[EGO] = UncertainOutcomeAgent(my_name=EGO,
#                                                  pref_distr=pref_distr,
#                                                  ego_pref=ego_pref,
#                                                  game_params=game_params_ego,
#                                                  world=traj_world,
#                                                  other_stopping_time=stopping_time,
#                                                  action_selection_method=selection_method)
#
#     return SimContext(
#         dg_scenario=DgScenario(scenario),
#         models=models,
#         players=players,
#         param=sim_params,
#     )


def get_scenario_4_way_crossing_uncertain_outcome_agent(
    pref_structures: Optional[Mapping[PlayerName, str]] = None,
    sim_params: Optional[SimParameters] = None,
    prob_go: float = 0.5,
    selection_method: Optional[str] = None,
    belief_distr: Optional[ProbDist] = None,
) -> SimContext:
    """
    Generate 4 way crossing scenario "DEU_Ffb-1_7_T-1" with an UncertainOutcomeAgent and a TrajectoryFollowing
    (new version). Trajectories are hand-crafted.
    """
    scenario_name = "DEU_Ffb-1_7_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)

    plot = False
    draw_labels = True

    # initial Dynamic Vehicle States
    x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2.0, vx=kmh2ms(15), delta=0)
    x0_ego = VehicleStateDyn(x=45.0, y=0.0, theta=0.0, vx=kmh2ms(25), delta=0)

    # vehicle dynamics and geometris
    p1_model = VehicleModelDyn.default_car(x0=x0_p1)
    vg_ego = VehicleGeometry.default_car(color="firebrick")
    vg_p1 = VehicleGeometry.default_car(color="firebrick")
    ego_model = VehicleModelDyn.default_car(x0_ego)
    ego_model.vg = vg_ego

    geos = {EGO: vg_ego, OTHER: vg_p1}

    models = {OTHER: p1_model, EGO: ego_model}

    # initial Vehicle States
    init_p1 = VehicleState(x=x0_p1.x, y=x0_p1.y, vx=x0_p1.vx, theta=x0_p1.theta, delta=x0_p1.delta)
    init_ego = VehicleState(x=x0_ego.x, y=x0_ego.y, vx=x0_ego.vx, theta=x0_ego.theta, delta=x0_ego.delta)

    initial_states = {P1: init_p1, EGO: init_ego}

    # optional plotting
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

    # set default simulation parameters
    if sim_params is None:
        sim_params = SimParameters(
            dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(3.9), max_sim_time=D(7.5)
        )

    if belief_distr is None:
        belief_over_p1 = ProbDist(
            {
                PosetalPreference("go_agent", use_cache=False): Fraction(1, 1),
                PosetalPreference("stop_agent", use_cache=False): Fraction(0, 1),
            }
        )

        pref_distr = {OTHER: belief_over_p1}

    else:
        pref_distr = {OTHER: belief_distr}

    if selection_method is None:
        selection_method = "avg"

    ego_pref = PosetalPreference("pref_intersection", use_cache=False)

    # refinement 1
    # ego_pref = PosetalPreference("pref_intersection_refined_1", use_cache=False)
    # refinement 2
    # ego_pref = PosetalPreference("pref_intersection_refined_2", use_cache=False)

    # trajectory generator
    params = TrajectoryGenParams(
        solve=False,
        s_final=-1,
        max_gen=7,
        dt=D("1.0"),
        u_acc=frozenset([-2.0, 2.0]),
        u_dst=frozenset([0.0]),
        v_max=ego_model.vp.vx_limits[1],
        v_min=0.0,
        st_max=ego_model.vp.delta_max,
        dst_max=ego_model.vp.ddelta_max,
        dt_samp=D("0.2"),
        dst_scale=False,
        n_factor=1.0,
        vg=VehicleGeometry.default_car(),
        v_switch=4.5,
        acc_max=11.5,
    )

    trajs_and_commands = {}
    ego_actions = get_stop_or_go_trajectories_horizontal(initial_state=init_ego, stopping_time=2.5, params=params)
    p1_actions = get_stop_or_go_trajectories_vertical(initial_state=init_p1, stopping_time=2.5, params=params)
    trajs_and_commands[EGO] = ego_actions

    # modify format of p1_actions
    p1_actions_mod = {}
    for key, dict_value in p1_actions.items():
        for traj, comm in dict_value.items():
            p1_actions_mod[traj] = comm

    p1_actions_mod = frozendict(p1_actions_mod)
    trajs_and_commands[PlayerName("Other")] = p1_actions_mod

    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
        P1: params,
        EGO: params,
    }

    net = scenario.lanelet_network
    ref_lanes: Mapping[PlayerName, List[RefLaneGoal]] = {}
    # compute all reference lanes
    for agent in models:
        x0 = models[agent].get_state()
        p = np.array([x0.x, x0.y])
        ref_lanes[agent] = [RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=1000)]

    traj_world = TrajectoryWorld(map_name=scenario_name, scenario=DgScenario(scenario), geo=geos, goals=ref_lanes)

    game_params_ego = TrajectoryGamePosetsParam(
        scenario=DgScenario(scenario),
        initial_states=initial_states,
        ref_lanes=ref_lanes,
        pref_structures=pref_structures,
        traj_gen_params=traj_gen_params,
        n_traj_max=10,
        sampling_method="uniform",
    )

    players = {}
    stopping_time = D(2.5)
    for agent in models:
        x0 = models[agent].get_state()
        if agent == OTHER:
            players[OTHER] = TrajectoryFollowingAgent(
                trajectory=list(p1_actions["go"].keys())[0],
                commands=list(p1_actions["go"].values())[0],
                alternative_trajectories={list(p1_actions["stop"].keys())[0]},
            )

        if agent == EGO:
            players[EGO] = UncertainOutcomeAgent(
                my_name=EGO,
                pref_distr=pref_distr,
                ego_pref=ego_pref,
                game_params=game_params_ego,
                world=traj_world,
                other_stopping_time=stopping_time,
                action_selection_method=selection_method,
                trajectories_and_commands=trajs_and_commands,
            )

    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        param=sim_params,
    )


# def get_scenario_4_way_crossing_uncertain_NE(pref_structures: Optional[Mapping[PlayerName, str]] = None,
#                                              sim_params: Optional[SimParameters] = None,
#                                              prob_go: float = 0.5,
#                                              selection_method: Optional[str] = None,
#                                              belief_distr: Optional[ProbDist] = None,
#                                              ) -> SimContext:
#     """
#     Generate 4 way crossing scenario "DEU_Ffb-1_7_T-1" with an StochasticGamePlayingAgent and a StopOrGoAgent
#     """
#     scenario_name = "DEU_Ffb-1_7_T-1"
#     scenario, planning_problem_set = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)
#
#     plot = False
#     draw_labels = True
#
#     # initial Dynamic Vehicle States
#     x0_p1 = VehicleStateDyn(x=70, y=-17.5, theta=pi / 2.0, vx=kmh2ms(20), delta=0)
#     x0_ego = VehicleStateDyn(x=45.0, y=0.0, theta=0.0, vx=kmh2ms(25), delta=0)
#
#     # vehicle dynamics and geometris
#     p1_model = VehicleModelDyn.default_car(x0=x0_p1)
#     vg_ego = VehicleGeometry.default_car(color="firebrick")
#     vg_p1 = VehicleGeometry.default_car(color="firebrick")
#     ego_model = VehicleModelDyn.default_car(x0_ego)
#     ego_model.vg = vg_ego
#
#     geos = {
#         EGO: vg_ego,
#         P1: vg_p1
#     }
#
#     models = {
#         P1: p1_model,
#         EGO: ego_model
#     }
#
#     # initial Vehicle States
#     init_p1 = VehicleState(x=x0_p1.x, y=x0_p1.y, vx=x0_p1.vx, theta=x0_p1.theta, delta=x0_p1.delta)
#     init_ego = VehicleState(x=x0_ego.x, y=x0_ego.y, vx=x0_ego.vx, theta=x0_ego.theta, delta=x0_ego.delta)
#
#     initial_states = {
#         P1: init_p1,
#         EGO: init_ego
#     }
#
#     # optional plotting
#     if plot:
#         matplotlib.use("TkAgg")
#         renderer: MPRenderer = MPRenderer()
#         renderer.draw_params["trajectory"]["draw_trajectory"] = False
#         renderer.draw_params["dynamic_obstacle"]["draw_shape"] = False
#         if draw_labels:
#             renderer.draw_params["lanelet"]["show_label"] = True
#
#         scenario.draw(renderer)
#         renderer.render()
#         plt.show()
#
#     # set default simulation parameters
#     if sim_params is None:
#         sim_params = SimParameters(dt=D("0.1"), dt_commands=D("0.1"), sim_time_after_collision=D(2),
#                                    max_sim_time=D(4.5))
#
#     ego_pref = PosetalPreference("pref_leon_dev_3", use_cache=False)
#
#     if belief_distr is None:
#         belief_over_p1 = ProbDist({
#             PosetalPreference("go_agent", use_cache=False): Fraction(1, 10),
#             PosetalPreference("stop_agent", use_cache=False): Fraction(9, 10)
#         })
#
#         pref_distr = {P1: belief_over_p1, EGO: ProbDist({ego_pref: Fraction(1, 1)})}
#
#     else:
#         pref_distr = {P1: belief_distr, EGO: ProbDist({ego_pref: Fraction(1, 1)})}
#
#     if selection_method is None:
#         selection_method = "avg"
#
#     # trajectory generator
#     params = TrajectoryGenParams(
#         solve=False,
#         s_final=-1,
#         max_gen=7,
#         dt=D("1.0"),
#         u_acc=frozenset([-2.0, 2.0]),
#         u_dst=frozenset([0.0]),
#         v_max=ego_model.vp.vx_limits[1],
#         v_min=0.0,
#         st_max=ego_model.vp.delta_max,
#         dst_max=ego_model.vp.ddelta_max,
#         dt_samp=D("0.2"),
#         dst_scale=False,
#         n_factor=1.0,
#         vg=VehicleGeometry.default_car(),
#         v_switch=4.5,
#         acc_max=11.5
#     )
#
#     traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
#         P1: params,
#         EGO: params,
#     }
#
#     net = scenario.lanelet_network
#     ref_lanes: Mapping[PlayerName, List[RefLaneGoal]] = {}
#     # compute all reference lanes
#     for agent in models:
#         x0 = models[agent].get_state()
#         p = np.array([x0.x, x0.y])
#         ref_lanes[agent] = [RefLaneGoal(dglane_from_position(p, net, succ_lane_selection=2), goal_progress=1000)]
#
#     traj_world = TrajectoryWorld(map_name=scenario_name, scenario=DgScenario(scenario), geo=geos, goals=ref_lanes)
#
#     game_params_ego = TrajectoryGamePosetsParam(
#         scenario=DgScenario(scenario),
#         initial_states=initial_states,
#         ref_lanes=ref_lanes,
#         pref_structures=pref_structures,
#         traj_gen_params=traj_gen_params,
#         n_traj_max=10,
#         sampling_method="uniform"
#     )
#
#     players = {}
#     stopping_time = D(2.5)
#     for agent in models:
#         x0 = models[agent].get_state()
#         p = np.array([x0.x, x0.y])
#         if agent == P1:
#             players[P1] = StopOrGoAgent(
#                 ref_lane=dglane_from_position(p, net, succ_lane_selection=2),
#                 prob_go=prob_go,
#                 behavior="go",  # force behavior to stop/go -> probability prob_go is ignored
#                 stopping_time=stopping_time,
#                 nominal_speed=init_p1.vx  # constant speed
#             )
#
#         if agent == EGO:
#             players[EGO] = StochasticGamePlayingAgent(my_name=EGO,
#                                                       pref_distr=pref_distr,
#                                                       game_params=game_params_ego,
#                                                       world=traj_world,
#                                                       other_stopping_time=stopping_time)
#
#     return SimContext(
#         dg_scenario=DgScenario(scenario),
#         models=models,
#         players=players,
#         param=sim_params,
#     )


def get_simulation_campaign_from_params(params: SimulationCampaignParams) -> List[SimContext]:
    """
    Generate a set of simulation contexts from "get_scenario_4_way_crossing_game_playing_agent()"
    """
    sim_contexts = []
    player_types = params.player_types.values()
    player_names = list(params.player_types.keys())
    for combination in product(*player_types):
        type_combination: Mapping[PlayerName, str] = {player_names[i]: combination[i] for i in range(len(player_names))}
        sim_contexts.append(
            get_scenario_4_way_crossing_game_playing_agent(
                pref_structures=type_combination,
                sim_params=params.sim_params,
                receding_horizon_time=params.receding_horizon_time,
            )
        )

    return sim_contexts


def get_scenario_4_way_crossing_stochastic_multiple_type_beliefs():
    """
    Test multiple times "get_scenario_4_way_crossing_game_playing_agent()" by varying the ego preference
    """
    EGO = PlayerName("Ego")
    P1 = PlayerName("P1")
    player_types: Mapping[PlayerName, List[str]] = {
        EGO: ["pref_leon_dev_4"],
        P1: ["pref_leon_dev", "pref_leon_dev_1", "pref_leon_dev_2", "pref_leon_dev_3", "pref_leon_dev_4"],
    }

    campaign_params: SimulationCampaignParams = SimulationCampaignParams(
        n_experiments=10, player_types=player_types  # for now not used -> use for statistics
    )
    sim_context_set = get_simulation_campaign_from_params(campaign_params)
    return sim_context_set
