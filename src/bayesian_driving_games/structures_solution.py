from typing import Mapping
from dataclasses import dataclass

from bayesian_driving_games.structures import PlayerType, BayesianGame, T
from games import (
    GameNode,
    SolvingContext,
    PlayerName,
    GamePreprocessed,
    X,
    U,
    Y,
    RP,
    RJ,
    SR,
    GamePlayerPreprocessed,
)

from possibilities import Poss

__all__ = ["BayesianGameNode", "BayesianSolvingContext", "BayesGamePreprocessed"]


@dataclass
class BayesianGameNode(GameNode):
    game_node_belief: Mapping[PlayerName, Poss[PlayerType]]
    """ Belief of each player at this game node """


@dataclass
class BayesianSolvingContext(SolvingContext):
    game: BayesianGame


@dataclass
class BayesGamePreprocessed(GamePreprocessed):
    """ A pre-processed Bayesian game. """

    game: BayesianGame[X, U, Y, RP, RJ, SR, T]
    """ The original game. """

    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]]
    """ The pre-processed data for each player"""
