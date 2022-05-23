import math
import os
import random
from typing import List, Mapping, FrozenSet

import matplotlib
import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics

from commonroad_challenge.situational_traj_generator import SituationalTrajectoryGenerator
from commonroad_challenge.utils import get_initial_states, get_default_pref_structures, generate_ref_lanes, \
    convert_from_cr_state, interacting_agents
from dg_commons import PlayerName, Timestamp, logger
from dg_commons.planning import Trajectory as Dg_Trajectory
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario
from trajectory_games import get_traj_game_posets_from_params, get_only_context, Solution, SolvedTrajectoryGame, \
    PosetalPreference, TrajectoryGenerator
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
        ego_vehicle_state: State,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        inter_agents: List[DynamicObstacle],
        time_to_goal: Timestamp) -> TrajectoryGamePosetsParam:

    initial_states: Mapping[PlayerName, VehicleState] = get_initial_states(ego_vehicle_state, inter_agents)

    ref_lanes, route_curvature_ego = generate_ref_lanes(
        scenario=scenario,
        planning_problem=planning_problem,
        inter_agents=inter_agents,
    )

    sit_traj_gens: Mapping[PlayerName, SituationalTrajectoryGenerator] = {
        PlayerName("Ego"): SituationalTrajectoryGenerator(ref_lane_goal=ref_lanes[PlayerName("Ego")],
                                                          path_curvature=route_curvature_ego
                                                          )
    }
    for obs in inter_agents:
       sit_traj_gens[PlayerName(str(obs.obstacle_id))] = \
           SituationalTrajectoryGenerator(ref_lane_goal=ref_lanes[PlayerName(str(obs.obstacle_id))])

    pref_structures: Mapping[PlayerName, str] = get_default_pref_structures(inter_agents)

    # todo: integrate SituationalTrajectoryGenerator better
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = {
        PlayerName("Ego"): sit_traj_gens[PlayerName("Ego")].get_traj_gen_params(
            state=convert_from_cr_state(ego_vehicle_state), time_to_goal=time_to_goal, is_ego=True
        )
    }
    for obs in inter_agents:
        obs_name = PlayerName(str(obs.obstacle_id))
        traj_gen_params[obs_name] =\
            sit_traj_gens[obs_name].get_traj_gen_params(
                state=convert_from_cr_state(obs.initial_state), time_to_goal=time_to_goal, is_ego=False
            )

    n_traj_max: Mapping[PlayerName, int] = {PlayerName("Ego"): 40}
    for dyn_obs in inter_agents:
        n_traj_max[PlayerName(str(dyn_obs.obstacle_id))] = 1

    return TrajectoryGamePosetsParam(
        scenario=DgScenario(scenario),
        initial_states=initial_states,
        ref_lanes=ref_lanes,
        pref_structures=pref_structures,
        traj_gen_params=traj_gen_params,
        n_traj_max=n_traj_max
    )
# todo: integrate Situational Trajectory Generator better
def filter_actions(trajectories: FrozenSet[Dg_Trajectory], n_actions: int = 10) -> FrozenSet[Dg_Trajectory]:
    """
    Filter actions through a set of criteria, e.g. feasibility
    :return:
    """
    vehicle_dynamics = VehicleDynamics.KS(VehicleType.FORD_ESCORT)

    subset_trajs = set()
    remaining_trajs = set(trajectories)

    while len(subset_trajs) < n_actions and len(remaining_trajs) != 0:
        cand_traj = random.sample(remaining_trajs, 1)[0]
        dt = cand_traj.timestamps[1] - cand_traj.timestamps[0]
        remaining_trajs.remove(cand_traj)
        # todo: account for feasibility
        # feasible = feasibility_check(cand_traj, vehicle_dynamics, dt)
        # if feasible:
        # print("found one feasible trajectory")
        subset_trajs.add(cand_traj)

    # print("Total number of trajectories: " + str(len(subset_trajs)))
    return frozenset(subset_trajs)


def generate_actions(
        trajectory_gens: Mapping[PlayerName, TrajectoryGenerator],
        initial_states: Mapping[PlayerName, VehicleState],
        max_n_traj: Mapping[PlayerName, int],
        non_empty_action_sets: bool = True) -> Mapping[PlayerName, FrozenSet[Dg_Trajectory]]:

    def create_const_traj(initial_state: VehicleState, duration: float, dt: float, axle_length: float):
        timestamps = [0.0]
        values = [initial_state]
        current_state = initial_state
        for t in np.arange(start=dt, stop=duration + dt, step=dt):
            v_next = current_state.vx
            delta_next = current_state.delta

            x_next = current_state.x + current_state.vx * math.cos(current_state.theta) * dt
            y_next = current_state.y + current_state.vx * math.sin(current_state.theta) * dt
            theta_next = current_state.theta + dt * math.tan(current_state.delta) * current_state.vx / axle_length

            current_state = VehicleState(x=x_next, y=y_next, theta=theta_next, delta=delta_next, vx=v_next)
            values.append(current_state)
            timestamps.append(t)
        return Dg_Trajectory(timestamps=timestamps, values=values)

    logger.info(f"Generating Trajectories for all players.")
    assert trajectory_gens.keys() == initial_states.keys(), "Player mismatch."
    all_trajs: Mapping[PlayerName, FrozenSet[Dg_Trajectory]] = {}
    for player_name, player_traj_gen in trajectory_gens.items():
        if player_name == PlayerName("Ego"):
            all_trajs[player_name] = player_traj_gen.get_actions(state=initial_states[player_name], return_graphs=False)
        else:
            #creating one constant action
            params = player_traj_gen.params
            init_state = initial_states[player_name]
            duration = (params.max_gen-1)*float(params.dt)
            dt = float(params.dt_samp)
            axle_length = params.vg.length
            all_trajs[player_name] = frozenset([create_const_traj(initial_state=init_state,
                                                                  duration=duration,
                                                                  dt=dt,
                                                                  axle_length=axle_length)])

    logger.info(f"Trajectory generation finished for all players.")

    subsampled_trajs: Mapping[PlayerName, FrozenSet[Dg_Trajectory]] = {}
    # subsample actions and filter
    for player_name, player_trajs in all_trajs.items():
        subset_trajs_p = filter_actions(trajectories=frozenset(player_trajs),
                                        n_actions=max_n_traj[player_name])
        subsampled_trajs[player_name] = frozenset(subset_trajs_p)

    # todo: do this also if an the action of a player is shorter than the one of other players
    # create an action of constant velocity and steering if no other actions are available

    if non_empty_action_sets:
        for player_name, traj_set in subsampled_trajs.items():
            if len(traj_set) == 0:
                p_traj_gen_params = trajectory_gens[player_name].params
                axle_length = p_traj_gen_params.vg.length
                init_state = initial_states[player_name]
                duration = float(p_traj_gen_params.dt) * (p_traj_gen_params.max_gen-1)
                dt = float(p_traj_gen_params.dt_samp)
                const_traj = create_const_traj(initial_state=init_state, duration=duration, dt=dt,
                                               axle_length=axle_length)
                # logger.info(const_traj)
                subsampled_trajs[player_name] = frozenset([const_traj])

    return subsampled_trajs


def compute_ego_trajectory(
        ego_vehicle_state: State,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        current_time: Timestamp,
        output_dir: str,
        current_timestep: int,
        total_time: Timestamp) -> Trajectory:

    inter_agents = interacting_agents(scenario=scenario,
                                      ego_state=ego_vehicle_state,
                                      look_ahead_dist=50.0,
                                      around_dist_r=5.0,
                                      around_dist_f=5.0,
                                      around_dist_lat=5.0
                                      )


    inter_agents = list(inter_agents.values())
    inter_agents_flat = [obs for obs_list in inter_agents for obs in obs_list]


    time_to_goal = total_time-current_time
    game_params = get_game_params(ego_vehicle_state, scenario, planning_problem, inter_agents_flat, time_to_goal)

    traj_gen: Mapping[PlayerName, TrajectoryGenerator] = {}
    for player_name, traj_gen_params in game_params.traj_gen_params.items():
        traj_gen[player_name] = TrajectoryGenerator(params=traj_gen_params, ref_lane_goals=[game_params.ref_lanes[player_name]])


    game = get_traj_game_posets_from_params(game_params=game_params)

    all_trajs = generate_actions(trajectory_gens=traj_gen, initial_states=game_params.initial_states,
                                 max_n_traj=game_params.n_traj_max, non_empty_action_sets=True)

    # create solving context and generate candidate trajectories for each agent

    solving_context = get_only_context(sgame=game, actions=all_trajs)


    sol: Solution = Solution()
    # compute NE
    nash_eqs: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)

    # select one NE at random between all the available admissible NE
    selected_eq = select_admissible_eq_randomly(eqs=nash_eqs)

    ####### PLOTTING CURRENT DECISION MAKING STEP AND OUTCOMES #######
    all_trajectories = solving_context.player_actions
    matplotlib.use("TkAgg")
    player_states: Mapping[PlayerName, VehicleState] = {}
    player_states[PlayerName("Ego")] = convert_from_cr_state(ego_vehicle_state)

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
