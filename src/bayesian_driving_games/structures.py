from dataclasses import dataclass
from typing import NewType, List, Mapping

from possibilities import Poss

from driving_games.structures import VehicleState
from games import GamePlayer, Game, PlayerName, GameSpec

__all__ = [
    "PlayerType",
    "AGGRESSIVE",
    "CAUTIOUS",
    "NEUTRAL",
    "BayesianGamePlayer",
    "BayesianVehicleState",
    "BayesianGame",
]

PlayerType = NewType("PlayerType", str)
""" The type of a player. """

AGGRESSIVE = PlayerType("aggressive")
""" aggressive player type optimizes for time. """

CAUTIOUS = PlayerType("cautious")
""" cautious player optimizes for comfort """

NEUTRAL = PlayerType("neutral")
"""Neutral player type"""


@dataclass
class BayesianGamePlayer(GamePlayer):
    # todo: Extend to more than two players - Mapping[PlayerName, List[PlayerType]]
    types_of_other: List[PlayerType]
    """The types of the other player"""

    types_of_myself: List[PlayerType]
    """The types of myself"""

    prior: Poss[PlayerType]
    """ The prior over the other player's types"""


# fixme az I suspect here we need Bayesian Game
@dataclass
class BayesianGame(Game):
    """ Definition of the game """

    players: Mapping[PlayerName, BayesianGamePlayer]
    """ The players in this game. """


@dataclass
class BayesianGameSpec(GameSpec):
    game: BayesianGame


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianVehicleState(VehicleState):
    player_type: PlayerType
    """ The type of the player """

    __print_order__ = ["x", "v", "player_type"]  # only print these attributes
