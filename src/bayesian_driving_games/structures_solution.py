from typing import Mapping
from dataclasses import dataclass

from bayesian_driving_games.structures import PlayerType
from games import GameNode, PlayerName
from possibilities import Poss


@dataclass
class BayesianGameNode(GameNode):
    game_node_belief: Mapping[PlayerName, Poss[PlayerType]]
    """ Belief of each player at this game node """
