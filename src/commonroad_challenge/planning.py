import os
from typing import List, Mapping

import matplotlib
import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from sumocr.interface.ego_vehicle import EgoVehicle

from commonroad_challenge.situational_traj_generator import SituationalTrajectoryGenerator
from commonroad_challenge.utils import get_initial_states, get_default_pref_structures, generate_ref_lanes, \
    convert_from_cr_state, interacting_agents
from dg_commons import PlayerName, Timestamp
from dg_commons.planning import Trajectory as Dg_Trajectory
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario
from trajectory_games import get_traj_game_posets_from_params, get_context_and_graphs, Solution, SolvedTrajectoryGame, \
    PosetalPreference
from trajectory_games.agents.game_playing_agent import select_admissible_eq_randomly
from trajectory_games.structures import TrajectoryGamePosetsParam, TrajectoryGenParams


def motion_planner_from_trajectory(trajectory: Dg_Trajectory, timestep: float) -> State:
    """
    Simply return Commonroad state from dg Trajectory at requested timestep, with interpolation.
    Caution: interpolation might violate solution feasibility
    """
    next_state = trajectory.at_interp(timestep)
    position = np.array([next_state.x, next_state.y])
    return State(position=position,
                 orientation=next_state.theta,
                 velocity=next_state.vx,
                 steering_angle=next_state.delta)


def get_game_params(
        ego_vehicle: EgoVehicle,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        inter_agents: List[DynamicObstacle],
        time_to_goal: Timestamp) -> TrajectoryGamePosetsParam:

    initial_states: Mapping[PlayerName, VehicleState] = get_initial_states(ego_vehicle, inter_agents)

    ref_lanes, route_curvature_ego = generate_ref_lanes(
        scenario,
        planning_problem,
        inter_agents
    )

    sit_traj_gens: Mapping[PlayerName, SituationalTrajectoryGenerator] = {
        PlayerName("Ego"): SituationalTrajectoryGenerator(ref_lane_goal=ref_lanes[PlayerName("Ego")], path_curvature=route_curvature_ego)
    }
    for obs in inter_agents:
       sit_traj_gens[PlayerName(str(obs.obstacle_id))] = \
           SituationalTrajectoryGenerator(ref_lane_goal=ref_lanes[PlayerName(str(obs.obstacle_id))])

    pref_structures: Mapping[PlayerName, str] = get_default_pref_structures(inter_agents)
    # traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = get_traj_gen_params(inter_agents)

    # todo: interate SituationalTrajectoryGenerator better
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
        PlayerName("Ego"): sit_traj_gens[PlayerName("Ego")].get_traj_gen_params(state=convert_from_cr_state(ego_vehicle.current_state), time_to_goal=time_to_goal, is_ego=True)
    }
    for obs in inter_agents:
        traj_gen_params[PlayerName(str(obs.obstacle_id))] = sit_traj_gens[PlayerName(str(obs.obstacle_id))].get_traj_gen_params(state=convert_from_cr_state(obs.initial_state), time_to_goal=time_to_goal, is_ego=False)

    # n_traj_max: int = 5
    n_traj_max: Mapping[PlayerName, int] = {}
    n_traj_max[PlayerName("Ego")] = 30
    for dyn_obs in inter_agents:
        n_traj_max[PlayerName(str(dyn_obs.obstacle_id))] = 1

    refresh_time: Timestamp = 0.5

    return TrajectoryGamePosetsParam(
        scenario=DgScenario(scenario),
        initial_states=initial_states,
        ref_lanes=ref_lanes,
        pref_structures=pref_structures,
        traj_gen_params=traj_gen_params,
        n_traj_max=n_traj_max,
        refresh_time=refresh_time
    )


def compute_ego_trajectory(
        ego_vehicle: EgoVehicle,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        current_time: Timestamp,
        output_dir: str,
        current_timestep: int,
        total_time: Timestamp = 15.0) -> Trajectory:
    inter_agents = interacting_agents(scenario=scenario,
                                      ego_state=ego_vehicle.current_state,
                                      look_ahead_dist=50.0,
                                      around_dist_r=5.0,
                                      around_dist_f=5.0,
                                      around_dist_lat=5.0
                                      )

    inter_agents = list(inter_agents.values())
    inter_agents_flat = [obs for obs_list in inter_agents for obs in obs_list]
    assert len(inter_agents) > 0, "There are no interacting agents."




    ######## FROM HERE WILL BE REPLACED BY STOCHASTIC DECISION MAKING#############

    time_to_goal = total_time-current_time
    game_params = get_game_params(ego_vehicle, scenario, planning_problem, inter_agents_flat, time_to_goal)

    game = get_traj_game_posets_from_params(game_params=game_params)

    # create solving context and generate candidate trajectories for each agent

    solving_context, _ = get_context_and_graphs(
        game=game,
        max_n_traj=game_params.n_traj_max,
        pad_trajectories=False,
        sampling_method=game_params.sampling_method
    )

    all_trajectories = solving_context.player_actions

    sol: Solution = Solution()
    # compute NE
    nash_eqs: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)

    # select one NE at random between all the available admissible NE
    selected_eq = select_admissible_eq_randomly(eqs=nash_eqs)

    ######## UNTIL HERE WILL BE REPLACED BY STOCHASTIC DECISION MAKING#############

    ####### PLOTTING CURRENT DECISION MAKING STEP AND OUTCOMES #######
    matplotlib.use("TkAgg")
    player_states: Mapping[PlayerName, VehicleState] = {}
    player_states[PlayerName("Ego")] = convert_from_cr_state(ego_vehicle.current_state)

    for dyn_obs in inter_agents_flat:
        player_states[PlayerName(str(dyn_obs.obstacle_id))] = convert_from_cr_state(dyn_obs.initial_state)

    output_folder = os.path.join(output_dir, "decision_steps")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # time_str = datetime.now().strftime("%y-%m-%d-%H%M%S")
    filename = scenario.scenario_id.map_name + "_" + str(current_timestep)
    file_path = os.path.join(output_folder, filename)

    game.game_vis.plot(
        player_states=player_states,
        player_actions=all_trajectories,
        player_refs=game.world.goals,
        player_eqs=selected_eq.actions,
        show_plot=False,
        filename=file_path
    )

    # output_folder = os.path.join(output_folder, "outcomes")
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    outcome_filename = "eq_outcomes_" + str(current_timestep)
    outcome_file_path = os.path.join(output_folder, outcome_filename)
    # plot outcomes and preferences for all players
    player_prefs: Mapping[PlayerName, PosetalPreference] = {}
    for player_name, game_player in game.game_players.items():
        player_prefs[player_name] = game_player.preference

    game.game_vis.plot_outcomes(
        player_outcomes=selected_eq.outcomes,
        player_prefs=player_prefs,
        title_info="step: " + str(current_timestep),
        filename=outcome_file_path
    )

    ####### PLOTTING CURRENT DECISION MAKING STEP AND OUTCOMES #######

    # get trajectory and commands relating to selected equilibria
    trajectory = selected_eq.actions[PlayerName("Ego")]

    return trajectory
