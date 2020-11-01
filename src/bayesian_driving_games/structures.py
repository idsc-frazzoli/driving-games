import itertools
from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from functools import lru_cache
from typing import FrozenSet, Mapping, NewType, List

from frozendict import frozendict

from driving_games import VehicleActions
from driving_games.structures import InvalidAction, LightsValue, VehicleState, VehicleDynamics
from games import Dynamics, GamePlayer
from possibilities import Poss
from zuper_commons.types import ZValueError
from driving_games.rectangle import Rectangle

__all__ = ["PlayerType", "BayesianVehicleState"]

PlayerType = NewType("PlayerType", str)
""" The type of a player. """

IN_A_RUSH = PlayerType("in a rush")
""" rushed player type optimizes for time. """

RELAXED = PlayerType("relaxed")
""" relaxed player optimizes for comfort """

NO_TYPE = PlayerType("no type")
""" No types assigned """


@dataclass
class BayesianGamePlayer(GamePlayer):
    types_of_other: List[PlayerType]
    """The types of the other player""" # TODO: Extend to more than two players - Mapping[PlayerName, List[PlayerType]]

    types_of_myself: List[PlayerType]
    """The types of myself"""

    prior: Poss[PlayerType]
    """ The prior over the other player's types"""


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianVehicleState(VehicleState):
    player_type: PlayerType
    """ The type of the player """

    __print_order__ = ["x", "v", "player_type"]  # only print these attributes

