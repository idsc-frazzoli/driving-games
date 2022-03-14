from typing import Mapping
from dg_commons import PlayerName
from trajectory_games import Solution, TrajectoryGame, SolvedTrajectoryGame, SolvingContext
from dataclasses import dataclass


# todo: might change structure here
# 1) there is only need for one world
# 2) monad?
# 3) only need for one visualization -> current implementation would not work

@dataclass
class DecentralizedTrajectoryGame:
    games: Mapping[PlayerName, TrajectoryGame]
    solving_contexts: Mapping[PlayerName, SolvingContext]
    nash_eqs: Mapping[PlayerName, Mapping[str, SolvedTrajectoryGame]]


    # todo this is easy attemp. Will need to remove visualizers from every payer's game and have a unique one
    def visualize_game(self):
        pass
