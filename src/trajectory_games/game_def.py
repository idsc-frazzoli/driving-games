import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Callable, TypeVar, Generic, FrozenSet, Optional, Tuple, Dict, Set, NewType

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
    "GameVisualization",
    "GamePlayer",
    "Game",
    "SolvedGameNode",
    "StaticSolverParams",
    "SolvingContext",
    "AntichainComparison",
    "EXP_ACCOMP",
    "JOIN_ACCOMP"
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


class ActionSetGenerator(Generic[X, U, W], ABC):
    """ A generic getter for the available actions """

    @abstractmethod
    def get_actions_static(self, state: X, player: PlayerName, world: W) -> FrozenSet[U]:
        pass


Key = TypeVar("Key")


class PlotStackDictionary(Generic[Key]):
    rows: int
    cols: int
    next_idx: int
    indices: Dict[Key, Tuple[int, int]]

    def __init__(self, values: Set[Key]):
        n_nodes = len(values)
        assert n_nodes > 0
        self.cols = math.ceil(n_nodes ** 0.5)
        self.rows = math.ceil(n_nodes / self.cols)
        self.next_idx = 0
        self.indices: Dict[Key, Tuple[int, int]] = {}

    def get_size(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def __getitem__(self, item: Key) -> Tuple[int, int]:
        if item in self.indices:
            return self.indices[item]
        if self.next_idx >= self.rows * self.cols:
            raise ValueError(f"Next idx = {self.next_idx}; rows,cols = {self.rows, self.cols}!")
        row = self.next_idx // self.cols
        col = self.next_idx % self.cols
        self.indices[item] = (row, col)
        self.next_idx += 1
        return row, col


class GameVisualization(Generic[X, U, W], ABC):
    """ A generic game visualization interface """
    plot_dict: Optional[PlotStackDictionary] = None

    @abstractmethod
    def plot_arena(self, axis):
        pass

    @abstractmethod
    def plot_player(self, axis, player_name: PlayerName, state: X, **kwargs):
        pass

    @abstractmethod
    def plot_actions(self, axis, actions: FrozenSet[U],
                     colour: Tuple[float, float, float], **kwargs):
        pass

    @abstractmethod
    def plot_equilibria(self, axis, actions: FrozenSet[U],
                        colour: Tuple[float, float, float], **kwargs):
        pass

    @abstractmethod
    def plot_pref(self, axis, pref: Preference[P], pname: PlayerName,
                  origin: Tuple[float, float],
                  labels: Mapping[str, str] = None):
        pass

    def init_plot_dict(self, values: Set[Key]):
        self.plot_dict = PlotStackDictionary[Key](values=values)

    def get_plot_dict(self) -> PlotStackDictionary:
        if self.plot_dict is None:
            raise ValueError(f"Initialise plot_dict first!")
        return self.plot_dict


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


AntichainComparison = NewType("AntichainComparison", str)
""" How to compare antichains. """
JOIN_ACCOMP = AntichainComparison("join_accomp")
EXP_ACCOMP = AntichainComparison("exp_accomp")


@dataclass(frozen=True)
class StaticSolverParams(SolverParams):
    admissible_strategies: AdmissibleStrategies
    """ Allowed search space of strategies"""
    strategy_multiple_nash: StrategyForMultipleNash
    """ How to deal with multiple Nash equilibria """
    antichain_comparison: AntichainComparison = EXP_ACCOMP
    """ How to compare antichains. """
    dt: Optional[Decimal] = None
    use_factorization: Optional[bool] = None
    use_best_response: bool = False
    """ Only use best responses for leader - follower game or all possible actions"""


@dataclass
class SolvingContext(Generic[X, U, W, P]):
    """ Context for the solution of the game"""

    player_actions: Mapping[PlayerName, FrozenSet[U]]
    """ All possible actions for each player"""

    game_outcomes: Callable[[JointPureActions], JointOutcome]
    """ The computed game outcomes. """

    outcome_pref: Mapping[PlayerName, Preference[P]]
    """ The preferences of each player. """

    solver_params: StaticSolverParams
    """Solver parameters"""

