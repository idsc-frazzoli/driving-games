from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Callable, TypeVar, Generic, FrozenSet, Optional, Tuple

from networkx import MultiDiGraph

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
    "ActionSetGenerator",
    "GameVisualization",
    "StaticGamePlayer",
    "StaticGame",
    "StaticSolvedGameNode",
    "StaticSolvingContext",
    "StaticSolverParams",
]

W = TypeVar("W")
"""Generic variable for a world"""
G = TypeVar("G")
"""Generic variable for a geometry"""

JointOutcome = Mapping[PlayerName, P]


class ActionSetGenerator(Generic[X, U, W], ABC):
    """ A generic getter for the available actions"""

    @abstractmethod
    def get_action_set(self, state: X, world: W, **kwargs) -> FrozenSet[U]:
        pass


class GameVisualization(Generic[X, U, W], ABC):
    """ A generic game visualization interface"""

    @abstractmethod
    def plot_arena(self, pylab, ax):
        pass

    @abstractmethod
    def plot_player(self, pylab, player_name: PlayerName, state: X):
        pass

    @abstractmethod
    def plot_actions(self, pylab, player: "StaticGamePlayer"):
        pass

    @abstractmethod
    def plot_equilibria(self, pylab, path: U, player: "StaticGamePlayer"):
        pass

    @abstractmethod
    def plot_pref(
        self, pylab, player: "StaticGamePlayer", origin: Tuple[float, float], labels: Mapping[str, str] = None
    ):
        pass


@dataclass
class StaticGamePlayer(Generic[X, U, W, P, G]):
    """ Information about one player. """

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
    graph: MultiDiGraph = None
    """The networkx graph"""


@dataclass
class StaticGame(Generic[X, U, W, P, G]):
    """A static game is a single stage game where players have a finite action set.
    We are considering games in which the agents live in a common world."""

    world: W
    """The world where the agents live"""
    game_players: Mapping[PlayerName, StaticGamePlayer[X, U, W, P, G]]
    """The players"""
    ps: PossibilityMonad
    """The game monad"""
    get_outcomes: Callable[[JointPureActions, W], JointOutcome]
    """The "game dynamics", given a pure action for each player, we have a distribution of outcomes"""
    game_vis: GameVisualization[X, U, W]
    """The game visualization"""


@dataclass(frozen=True, unsafe_hash=True)
class StaticSolvedGameNode(Generic[U, P]):
    """ Solved node of the game"""

    # todo this does not seem right
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
class StaticSolvingContext(Generic[X, U, W, P]):
    """ Context for the solution of the game"""

    player_actions: Mapping[PlayerName, FrozenSet[U]]
    """ All possible actions for each player"""

    game_outcomes: Mapping[JointPureActions, Poss[JointOutcome]]
    # TODO: Should we switch from Poss over joint outcomes to poss over outcomes?
    # Mapping[JointPureActions, Poss[Mapping[PlayerName, P]]] -->
    # Mapping[JointPureActions, Mapping[PlayerName, Poss[P]]]
    """ The computed game outcomes. """

    outcome_pref: Mapping[PlayerName, Preference[P]]
    """ The preferences of each player. """

    solver_params: StaticSolverParams
    """Solver parameters"""
