import random
from itertools import product
from typing import Optional, Mapping, FrozenSet, Any

from dg_commons import U, PlayerName, logger, DgSampledSequence, Timestamp
from dg_commons.planning.trajectory import Trajectory, TrajectoryGraph
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations
from dg_commons.time import time_function
from trajectory_games import Solution, SolvedTrajectoryGame, SolvedGameNode
from trajectory_games.game_factory import get_traj_game_posets_from_params
from trajectory_games.structures import TrajectoryGamePosetsParam
from trajectory_games.trajectory_game import get_context_and_graphs

__all__ = ["GamePlayingAgent"]


def select_admissible_eq_randomly(eqs: Mapping[str, SolvedTrajectoryGame]):
    admissible_eq = list(eqs['admissible'])
    if len(admissible_eq) > 1:
        logger.info(
            f"Randomly selecting one admissible equilibria out of "
            f"{len(admissible_eq)}" f" available equilibria."
        )
    return random.choice(admissible_eq)


class GamePlayingAgent(Agent):
    """Agent solving a trajectory game with posetal preferences, implemented in Receding Horizon.
    The plotting is implemented for a two player game"""

    def __init__(
            self,
            game_params: TrajectoryGamePosetsParam
    ):

        self.game_params = game_params
        self.my_name: Optional[PlayerName] = None
        # for plotting
        self.all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
        self.selected_eq: Optional[SolvedGameNode] = None
        # for control
        self.commands: Optional[DgSampledSequence[VehicleCommands]] = None
        self.trajectory: Optional[Trajectory] = None
        self.pseudo_start_time: Timestamp = 0.0

    def full_game_function(self):
        game = get_traj_game_posets_from_params(self.game_params)
        # create solving context and generate candidate trajectories for each agent
        solving_context, traj_graphs = get_context_and_graphs(
            game=game,
            max_n_traj=self.game_params.n_traj_max,
        )
        self.all_trajectories = solving_context.player_actions
        sol: Solution = Solution()
        # compute NE
        nash_eqs: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)
        # select one NE at random between all the available admissible NE
        self.selected_eq = select_admissible_eq_randomly(eqs=nash_eqs)
        # get trajectory and commands relating to selected equilibria
        self.trajectory = self.selected_eq.actions[self.my_name]
        # caution needed: only works when a single graph is passed
        my_graph: TrajectoryGraph = list(traj_graphs[self.my_name])[0]
        self.commands = my_graph.commands_on_trajectory(self.trajectory)

        # shift trajectory when receding horizon control is used
        self.trajectory = self.trajectory.shift_timestamps(self.pseudo_start_time)
        self.commands = self.commands.shift_timestamps(self.pseudo_start_time)

    @time_function
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.my_name = init_sim_obs.my_name
        # random.seed(init_sim_obs.seed)
        self.full_game_function()

    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        # solve game in receding horizon if a refresh_time is given
        if self.game_params.refresh_time and abs(current_time) > 0.0:
            new_initial_states: Mapping[PlayerName, VehicleState] = {}
            if abs(float(current_time) % self.game_params.refresh_time) == 0.0:
                self.pseudo_start_time = self.pseudo_start_time + self.game_params.refresh_time
                # change initial states for new game
                for pname, player_obs in sim_obs.players.items():
                    # convert from VehicleStateDyn to VehicleState
                    s_dyn = player_obs.state
                    new_initial_states[pname] = VehicleState(x=s_dyn.x, y=s_dyn.x, vx=s_dyn.vx,
                                                             theta=s_dyn.theta, delta=s_dyn.delta)

                self.game_params.initial_states = new_initial_states
                # generate new trajectories, process new game, solve game, compute trajectories and new commands
                self.full_game_function()

        if current_time < self.commands.get_start():
            logger.info('Warning, no commands defined so early. Returning first command input.')
            return self.commands.values[0]
        elif current_time > self.commands.get_end():
            logger.info('Warning, no commands defined so late. Returning last command input.')
            return self.commands.values[-1]
        else:
            return self.commands.at_or_previous(current_time)

    def on_get_extra(self) -> Optional[Any]:
        # store trajectories in extra of player logger (for plotting)
        P1 = PlayerName('P1')
        trajectories = self.all_trajectories[self.my_name]
        trajectories_blue = self.all_trajectories[P1]
        selected_trajectory_blue = self.selected_eq.actions[P1]
        selected_traj = self.trajectory
        candidates = tuple(
            product(
                trajectories,
                [
                    "indianred",
                ],
            )
        )
        new_tuple_red = (selected_traj, 'darkred')
        candidates += (new_tuple_red,)

        candidates_blue = tuple(
            product(
                trajectories_blue,
                [
                    "cornflowerblue",
                ],
            )
        )
        candidates += candidates_blue

        new_tuple_blue = (selected_trajectory_blue, 'mediumblue')
        candidates += (new_tuple_blue,)

        return candidates
