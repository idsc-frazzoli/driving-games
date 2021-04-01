from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Callable, TypeVar, Generic, FrozenSet, Optional, Tuple

from games import (
    PlayerName,
    U,
    X,
    P,
    JointPureActions,
    MonadicPreferenceBuilder,
    AdmissibleStrategies,
    StrategyForMultipleNash,
    SolverParams,
)
from possibilities import Poss, PossibilityMonad
from preferences import Preference

__all__ = [
    "W",
    "G",
    "U_all",
    "ActionGraph",
    "ActionSetGenerator",
    "StaticActionSetGenerator",
    "DynamicActionSetGenerator",
    "GameVisualization",
    "GamePlayer",
    "StaticGamePlayer",
    "DynamicGamePlayer",
    "Game",
    "SolvedGameNode",
    "StaticSolverParams",
    "SolvingContext",
    "StaticSolvingContext",
    "DynamicSolvingContext",
]

W = TypeVar("W")
"""Generic variable for a world"""
G = TypeVar("G")
"""Generic variable for a geometry"""
U_all = TypeVar("U_all")
"""Generic variable for all possible actions"""

JointOutcome = Mapping[PlayerName, P]


class ActionGraph(Generic[U], ABC):
    """ Dynamic action graph"""


class ActionSetGenerator(Generic[X, U, U_all, W], ABC):
    """ A generic getter for the available actions """

    @abstractmethod
    def get_actions_static(self, state: X, player: PlayerName, world: W) -> U_all:
        pass

    @abstractmethod
    def get_actions_dynamic(self, state: X, player: PlayerName, world: W) -> U_all:
        pass


class StaticActionSetGenerator(Generic[X, U, W], ActionSetGenerator[X, U, FrozenSet[U], W], ABC):
    """ A generic getter for the available static actions """

    @abstractmethod
    def get_actions_static(self, state: X, player: PlayerName, world: W) -> FrozenSet[U]:
        pass

    def get_actions_dynamic(self, state: X, player: PlayerName, world: W) -> U_all:
        raise NotImplementedError("Dynamic actions called for static generator")


class DynamicActionSetGenerator(Generic[X, U, W], ActionSetGenerator[X, U, ActionGraph[U], W], ABC):
    """ A generic getter for the dynamic action graph """

    @abstractmethod
    def get_actions_dynamic(self, state: X, player: PlayerName, world: W) -> ActionGraph[U]:
        pass

    def get_actions_static(self, state: X, player: PlayerName, world: W) -> FrozenSet[U]:
        raise NotImplementedError("Static actions called for dynamic generator")


class GameVisualization(Generic[X, U, W], ABC):
    """ A generic game visualization interface """

    @abstractmethod
    def plot_arena(self, pylab, ax):
        pass

    @abstractmethod
    def plot_player(self, pylab, player_name: PlayerName, state: X, **kwargs):
        pass

    @abstractmethod
    def plot_actions(self, pylab, player: "GamePlayer"):
        pass

    @abstractmethod
    def plot_equilibria(self, pylab, path: U, player: "GamePlayer"):
        pass

    @abstractmethod
    def plot_pref(self, pylab, player: "GamePlayer",
                  origin: Tuple[float, float],
                  labels: Mapping[str, str] = None):
        pass


@dataclass
class GamePlayer(Generic[X, U, W, P, G]):
    """ Information about one player. """

    name: PlayerName
    """The player's name"""
    state: Poss[X]
    """The player state in the world"""
    actions_generator: ActionSetGenerator
    """ Player dynamics """
    preference: Preference[JointOutcome]
    """ Its preferences about the outcomes. """
    monadic_preference_builder: MonadicPreferenceBuilder
    """ How to elevate and evaluate preferences over monadic outcomes."""
    vg: G
    """ Vehicle geometry """


@dataclass
class StaticGamePlayer(GamePlayer[X, U, W, P, G]):
    """ One player for static game. """

    actions_generator: StaticActionSetGenerator


@dataclass
class DynamicGamePlayer(GamePlayer[X, U, W, P, G]):
    """ One player for dynamic game. """

    actions_generator: DynamicActionSetGenerator


@dataclass
class Game(Generic[X, U, W, P, G]):
    """A game is a single/ multi stage game.
    The agents live in a common world."""

    world: W
    """The world where the agents live"""
    game_players: Mapping[PlayerName, GamePlayer[X, U, W, P, G]]
    """The players"""
    ps: PossibilityMonad
    """The game monad"""
    get_outcomes: Callable[[JointPureActions], JointOutcome]
    """The "game dynamics", given a pure action for each player, we have a distribution of outcomes"""
    game_vis: GameVisualization[X, U, W]
    """The game visualization"""


@dataclass(frozen=True, unsafe_hash=True)
class SolvedGameNode(Generic[U, P]):
    """ Solved node of the game"""

    actions: JointPureActions
    """ The final converged equilibrium actions """

    outcomes: JointOutcome
    """ Outcomes for each player """


@dataclass(frozen=True)
class StaticSolverParams(SolverParams):
    admissible_strategies: AdmissibleStrategies
    """ Allowed search space of strategies"""
    strategy_multiple_nash: StrategyForMultipleNash
    """ How to deal with multiple Nash equilibria """
    dt: Optional[Decimal] = None
    use_factorization: Optional[bool] = None


@dataclass
class SolvingContext(Generic[X, U, U_all, W, P]):
    """ Context for the solution of the game"""

    player_actions: Mapping[PlayerName, U_all]
    """ All possible actions for each player"""

    game_outcomes: Callable[[JointPureActions], JointOutcome]
    """ The computed game outcomes. """

    outcome_pref: Mapping[PlayerName, Preference[P]]
    """ The preferences of each player. """

    solver_params: StaticSolverParams
    """Solver parameters"""


@dataclass
class StaticSolvingContext(SolvingContext[X, U, FrozenSet[U], W, P]):
    """ Context for the solution of the static game"""


@dataclass
class DynamicSolvingContext(SolvingContext[X, U, ActionGraph[U], W, P]):
    """ Context for the solution of the dynamic game"""

