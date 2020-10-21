from typing import Mapping, Set
from dataclasses import dataclass

from bayesian_driving_games import PlayerType, BayesianGame
from games import GameNode, PlayerName, SolvingContext, JointState
from games.game_def import JointPureActions
from possibilities import Poss


@dataclass
class BayesianGameNode(GameNode):
    game_node_belief: Mapping[PlayerName, Poss[PlayerType]]
    """ Belief of each player at this game node """


InformationSet = Set[JointState]
#
# PlayerBelief = Poss[InformationSet]


@dataclass
class BayesianSolvingContext(SolvingContext):
    game: BayesianGame
    proposed_strategy: Mapping[GameNode, JointPureActions]
