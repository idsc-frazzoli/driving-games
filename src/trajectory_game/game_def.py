from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Callable, TypeVar, Generic, FrozenSet

from games import PlayerName, U, X, P, JointPureActions,  MonadicPreferenceBuilder
from possibilities import Poss, PossibilityMonad
from preferences import Preference

__all__ = [
    "W",
    "ActionSetGenerator",
    "StaticGamePlayer",
    "StaticGame",
    "StaticSolvedGameNode",
    "StaticSolvingContext",
]

W = TypeVar("W")
"""Generic variable for a world"""

JointOutcome = Mapping[PlayerName, P]


class ActionSetGenerator(Generic[X, U, W], ABC):

    @abstractmethod
    def get_action_set(self, state: X, world: W, player: PlayerName) -> FrozenSet[U]:
        """ Generate all possible actions for a given state and world. """


@dataclass
class StaticGamePlayer(Generic[X, U, W, P]):
    """ Information about one player. """

    name: PlayerName
    """The player's name"""
    state: Poss[X]
    """The player state in the world"""
    action_set_generator: ActionSetGenerator[X, U, W]
    """ Player dynamics """
    preferences: Preference[P]
    """ Its preferences about the combined joint/personal rewards. """
    monadic_preference_builder: MonadicPreferenceBuilder
    """ How to evaluate preferences over monadic outcomes. """


@dataclass
class StaticGame(Generic[X, U, W, P]):
    """ A static game is a single stage game where players have a finite action set.
    We are considering games in which the agents live in a common world. """

    world: W
    """The world where the agents live"""
    game_players: Mapping[PlayerName, StaticGamePlayer[X, U, W, P]]
    """The players"""
    ps: PossibilityMonad
    """How to evaluate stuff over monads"""
    game_outcomes: Callable[[JointPureActions, W], JointOutcome]
    """The "game dynamics", given a pure action for each player, we have a distribution of outcomes"""


@dataclass(frozen=True, unsafe_hash=True)
class StaticSolvedGameNode(Generic[U, P]):
    """ Solved node of the game"""

    actions: JointPureActions
    """ The final converged equilibrium actions """

    outcomes: JointOutcome
    """ Outcomes for each player """


@dataclass
class StaticSolvingContext(Generic[X, U, W, P]):
    """ Context for the solution of the game"""

    player_actions: Mapping[PlayerName, FrozenSet[U]]
    """ All possible actions for each player"""

    # TODO[SIR]: Extend to Poss
    game_outcomes: Mapping[JointPureActions, JointOutcome]
    """ The computed game outcomes. """

    outcome_pref: Mapping[PlayerName, Preference[P]]
    """ The preferences of each player. """
