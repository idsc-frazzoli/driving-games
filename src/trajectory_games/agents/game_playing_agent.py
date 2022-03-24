import random
from itertools import product
from typing import Optional, Mapping, FrozenSet

from dg_commons import U, PlayerName, logger, DgSampledSequence
from dg_commons.planning.trajectory import Trajectory, TrajectoryGraph
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.simulator_structures import SimObservations
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
            f"{len(admissible_eq)}" f"available equilibria."
        )
    return random.choice(admissible_eq)


class GamePlayingAgent(Agent):
    """Agent solving a trajectory game"""

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
        random.seed(0)  # todo: get this from Simulation Context when ready

    @time_function
    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

        game = get_traj_game_posets_from_params(self.game_params)
        # create solving context and generate candidate trajectories for each agent
        solving_context, traj_graphs = get_context_and_graphs(game=game)
        self.all_trajectories = solving_context.player_actions
        sol: Solution = Solution()
        # compute NE
        nash_eqs: Mapping[str, SolvedTrajectoryGame] = sol.solve_game(context=solving_context)
        # select one NE at random between all the available admissible NE
        self.selected_eq = select_admissible_eq_randomly(eqs=nash_eqs)

        # get trajectory and commands relating to selected equilibria
        self.trajectory = self.selected_eq.actions[self.my_name]
        # caution needed:  only works when a single graph is passe
        my_graph: TrajectoryGraph = list(traj_graphs[self.my_name])[0]
        self.commands = my_graph.commands_on_trajectory(self.trajectory)

    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        if current_time < self.commands.get_start():
            logger.info('Warning, no commands defined so early. Returning first command input.')
            return self.commands.values[0]
        elif current_time > self.commands.get_end():
            logger.info('Warning, no commands defined so late. Returning last command input.')
            return self.commands.values[-1]
        else:
            return self.commands.at_or_previous(
                current_time)  # todo: strange: at_interp is better at following a trajectory (only in one case)

    def on_get_extra(self) -> Optional[DrawableTrajectoryType]:
        trajectories = self.all_trajectories[self.my_name]
        trajectories_blue = self.all_trajectories[PlayerName('P1')]
        selected_traj = self.trajectory
        candidates = tuple(
            product(
                trajectories,
                [
                    "lightcoral",
                ],
            )
        )
        new_tuple = (selected_traj, 'red')
        candidates += (new_tuple,)

        candidates_blue = tuple(
            product(
                trajectories_blue,
                [
                    "blue",
                ],
            )
        )
        candidates += candidates_blue

        return candidates
