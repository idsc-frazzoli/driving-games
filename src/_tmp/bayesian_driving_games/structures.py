from dataclasses import dataclass, field
from typing import NewType, List, Mapping, TypeVar

from dg_commons import PlayerName
from driving_games.structures import VehicleState
from games.game_def import *
from possibilities import Poss

__all__ = [
    "PlayerType",
    "AGGRESSIVE",
    "CAUTIOUS",
    "NEUTRAL",
    "BayesianGamePlayer",
    "BayesianVehicleState",
    "BayesianGame",
    "T",
]

T = TypeVar("T")
""" Generic variable for the type of a player. """

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
    types_of_myself: Poss[PlayerType]
    """The types of myself"""

    prior: Mapping[PlayerName, Poss[PlayerType]]
    """ The prior over the other player's types"""

    types_of_others: Mapping[PlayerName, List[PlayerType]] = field(init=False)
    """The types of the other player"""

    def __post_init__(
        self,
    ):
        self.types_of_others = {k: poss.support for k, poss in self.prior.items()}


@dataclass
class BayesianGame(Game):
    """Definition of the Bayesian game"""

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
