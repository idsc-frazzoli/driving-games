from typing import Mapping
from dataclasses import dataclass

from _tmp.bayesian_driving_games.structures import PlayerType, BayesianGame
from games import (
    GameNode,
    SolvingContext,
    PlayerName,
    GamePreprocessed,
    GamePlayerPreprocessed,
    JointState,
)
from games.solve.solution_structures import GameGraph

from possibilities import Poss

__all__ = ["BayesianGameNode", "BayesianGameGraph", "BayesianSolvingContext", "BayesianGamePreprocessed"]


@dataclass
class BayesianGameNode(GameNode):
    game_node_belief: Mapping[PlayerName, Mapping[PlayerName, Poss[PlayerType]]]
    """ Belief of each player at this game node over the possible types of the other players"""


@dataclass
class BayesianGamePreprocessed(GamePreprocessed):
    """A pre-processed Bayesian game."""

    game: BayesianGame
    """ The original game. """

    players_pre: Mapping[PlayerName, GamePlayerPreprocessed]
    """ The pre-processed data for each player"""


@dataclass
class BayesianGameGraph(GameGraph):
    """The bayesian game graph."""

    state2node: Mapping[JointState, BayesianGameNode]


@dataclass
class BayesianSolvingContext(SolvingContext):
    """The bayesian solving context."""

    game: BayesianGame
    gg: BayesianGameGraph
