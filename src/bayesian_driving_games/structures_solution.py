from typing import Mapping, Generic, Set
from dataclasses import dataclass

from bayesian_driving_games import PlayerType
from games import GameNode, PlayerName, SolvingContext
from games.game_def import JointPureActions, JointState
from possibilities import Poss


@dataclass
class BayesianGameNode(GameNode):
    game_node_belief: Mapping[PlayerName, Poss[PlayerType]]
    """ Belief of each player at this game node """


InformationSet = Set[JointState]


PlayerBelief = Poss[InformationSet]


@dataclass
class BayesianSolvingContext(SolvingContext):
    proposed_strategy: Mapping[GameNode, JointPureActions]
