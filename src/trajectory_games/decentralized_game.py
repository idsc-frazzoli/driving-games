from typing import Mapping, Any
from dg_commons import PlayerName, Timestamp
from dg_commons.planning import Trajectory
from trajectory_games import Solution, TrajectoryGame, SolvedTrajectoryGame, SolvingContext, TrajectoryWorld
from dataclasses import dataclass


# todo: might change structure here
# 1) there is only need for one world
# 2) monad?
# 3) only need for one visualization -> current implementation would not work

@dataclass
class RecedingHorizonParams:
    N: int
    """Number of loops of the game to solve"""
    dt: Timestamp
    """Receding horizon timestep"""


    @staticmethod
    def default():
        return RecedingHorizonParams(N=3, dt=1.)

@dataclass
class DecentralizedTrajectoryGame:
    """
    Decentralized version of a trajectory game. Each player has its own instance of a Trajectory Game and
    computes its own nash equilibria.
    The world is shared among
    """
    games: Mapping[PlayerName, TrajectoryGame]
    solving_contexts: Mapping[PlayerName, SolvingContext]
    nash_eqs: Mapping[PlayerName, Mapping[str, SolvedTrajectoryGame]]

    def __post_init__(self):
        world = None
        for player, traj_game in self.games.items():
            if not world:
                world = traj_game.world
                continue
            #assert world == traj_game.world, "All players must live in the same world. "
            # todo[LEON]: define __eq__ method for world, or use one shared world





    # todo this is easy attemp. Will need to remove visualizers from every player's game and have a unique one
    def visualize_game(self):
        pass


History = Mapping[PlayerName, Trajectory]


@dataclass
class RecedingHorizonGame_draft: # todo [LEON]: for now only use to store game history and params
    #game: Any  # game should have a solve method
    history: History
    params: RecedingHorizonParams = RecedingHorizonParams.default()
