import os
from datetime import datetime
from typing import List, Mapping

import matplotlib
import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from sumocr.interface.ego_vehicle import EgoVehicle

from commonroad_challenge.utils import get_initial_states, get_default_pref_structures, generate_basic_refs, \
    get_traj_gen_params, convert_from_cr_state, interacting_agents
from dg_commons import PlayerName, Timestamp
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario
from trajectory_games import get_traj_game_posets_from_params, get_context_and_graphs, Solution, SolvedTrajectoryGame
from trajectory_games.agents.game_playing_agent import select_admissible_eq_randomly
from trajectory_games.structures import TrajectoryGamePosetsParam, TrajectoryGenParams


# todo 9: ugly. We are using dg_commons trajectory but the typehint is for commonroad Trajectory
def motion_planner_from_trajectory(trajectory: Trajectory, timestep: float) -> State:
    next_state = trajectory.at_interp(timestep)
    position = np.array([next_state.x, next_state.y])
    return State(position=position,
                 orientation=next_state.theta,
                 velocity=next_state.vx,
                 steering_angle=next_state.delta)


def get_game_params(ego_vehicle: EgoVehicle,
                    scenario: Scenario,
                    planning_problem: PlanningProblem,
                    interacting_agents: List[DynamicObstacle]) -> TrajectoryGamePosetsParam:
    initial_states: Mapping[PlayerName, VehicleState] = get_initial_states(ego_vehicle, scenario, interacting_agents)
    ref_lanes: Mapping[PlayerName, RefLaneGoal] = generate_basic_refs(ego_vehicle, scenario, planning_problem,
                                                                      interacting_agents)
    pref_structures: Mapping[PlayerName, str] = get_default_pref_structures(interacting_agents)
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams] = get_traj_gen_params(interacting_agents)
    n_traj_max: int = 5
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


def compute_ego_trajectory(ego_vehicle: EgoVehicle, scenario: Scenario,
                           planning_problem: PlanningProblem) -> Trajectory:
    # inter_agents = interacting_lateral(scenario, ego_vehicle.current_state)
    # inter_leading = interacting_leading(scenario, ego_vehicle.current_state, look_forward_dist=20.0)
    inter_agents = interacting_agents(scenario=scenario,
                                      ego_state=ego_vehicle.current_state,
                                      look_ahead_dist=30.0,
                                      around_dist_r=5.0,
                                      around_dist_f=5.0,
                                      around_dist_lat=5.0
                                      )

    inter_agents = list(inter_agents.values())
    inter_agents_flat = [obs for obs_list in inter_agents for obs in obs_list]
    assert len(inter_agents) > 0, "There are no interacting agents."

    game_params = get_game_params(ego_vehicle, scenario, planning_problem, inter_agents_flat)

    game = get_traj_game_posets_from_params(game_params=game_params)

    # create solving context and generate candidate trajectories for each agent

    solving_context, _ = get_context_and_graphs(
        game=game,
        max_n_traj=game_params.n_traj_max,
        pad_trajectories=True,
        sampling_method=game_params.sampling_method
    )

    all_trajectories = solving_context.player_actions

    sol: Solution = Solution()
    # compute NE
    nash_eqs: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)

    # select one NE at random between all the available admissible NE
    selected_eq = select_admissible_eq_randomly(eqs=nash_eqs)
    # plot current game for inspection
    matplotlib.use("TkAgg")
    player_states: Mapping[PlayerName, VehicleState] = {}
    player_states[PlayerName("Ego")] = convert_from_cr_state(ego_vehicle.current_state)
    for dyn_obs in inter_agents_flat:
        player_states[PlayerName(str(dyn_obs.obstacle_id))] = convert_from_cr_state(
            dyn_obs.initial_state)  # todo: here need to check at correct time, not at initial time

    filename = scenario.scenario_id.map_name + datetime.now().strftime("%y-%m-%d-%H%M%S")  # + ".pdf"
    output_folder = "/media/leon/Extreme SSD1/MT/outputs_TEST/decision_steps"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, filename)

    game.game_vis.plot(
        player_states=player_states,
        player_actions=all_trajectories,
        player_refs=game.world.goals,
        player_eqs=selected_eq.actions,
        player_outcomes=selected_eq.outcomes,
        show_plot=False,
        filename=file_path
    )
    # get trajectory and commands relating to selected equilibria
    trajectory = selected_eq.actions[PlayerName("Ego")]

    return trajectory
    # todo: add these costs for each metric in the pref structure on the report
    # compute metric violations for statistics
    # self.metric_violation.append(solving_context.game_outcomes(self.selected_eq.actions))
    # self.metric_violation.append(self.selected_eq.outcomes)  # todo: fix type?

    # shift trajectory when receding horizon control is used
    # trajectory = trajectory.shift_timestamps(pseudo_start_time)
    # commands = commands.shift_timestamps(pseudo_start_time)
