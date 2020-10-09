from typing import Mapping, Generic
from dataclasses import dataclass

from games import GameNode, PlayerName, SolvingContext
from games.game_def import PlayerBelief, JointPureActions


@dataclass
class BayesianGameNode(GameNode):
    belief: Mapping[PlayerName, PlayerBelief]
    """ Belief of each player at this game node """


@dataclass
class BayesianSolvingContext(SolvingContext):
    proposed_strategy: Mapping[GameNode, JointPureActions]