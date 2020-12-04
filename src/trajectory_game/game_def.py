from dataclasses import dataclass
from typing import Mapping, Callable, TypeVar, Generic, Set

from games import PlayerName, JointPureActions, U, X, MonadicPreferenceBuilder
from games.game_def import P
from possibilities import Poss
from preferences import Preference

W = TypeVar("W")
"""Generic variable for a world"""
ActionSetGenerator = Callable[[X, W], Set[U]]


@dataclass
class OpenLoopGamePlayer(Generic[X, U, W]):
    """ Information about one player. """

    state: Poss[X]
    """The player state in the world"""
    action_set_generator: ActionSetGenerator
    """ Player dynamics """
    preferences: Preference[P]
    """ Its preferences about the combined joint/personal rewards. """
    monadic_preference_builder: MonadicPreferenceBuilder
    """ How to evaluate preferences over monadic outcomes. """


@dataclass
class OpenLoopGame(Generic[X, U, W, P]):
    """An open loop game is a game where players have a finite action set.
    We are considering games in which the agent live in a common world and generate"""

    world: W
    """The world where the agents live"""
    game_players: Mapping[PlayerName, OpenLoopGamePlayer]
    """The players"""
    game_outcomes: Callable[[JointPureActions, W], Poss[P]]
    """The "game dynamics", given a pure action for each player, we have a distribution of outcomes"""
