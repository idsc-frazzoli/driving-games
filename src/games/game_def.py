from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from typing import (
    Callable,
    FrozenSet,
    Generic,
    Mapping,
    NewType,
    Optional,
    Tuple,
    TypeVar,
    Union, List,
)

from frozendict import frozendict

from possibilities import Poss, PossibilityMonad
from preferences import Preference
from zuper_commons.types import check_isinstance, ZValueError
from . import GameConstants

__all__ = [
    "Dynamics",
    "Y",
    "X",
    "U",
    "RP",
    "RJ",
    "Observations",
    "JointState",
    "JointPureActions",
    "JointMixedActions",
    "JointRewardStructure",
    "PersonalRewardStructure",
    "PlayerName",
    "Combined",
    "MonadicPreferenceBuilder",
    "Game",
    "GamePlayer",
    "GameVisualization",
]

PlayerName = NewType("PlayerName", str)
""" Strings that represent player's names/IDs. """

X = TypeVar("X")
""" Generic variable for a player's state."""

U = TypeVar("U")
""" Generic variable for a player's commands. """

Y = TypeVar("Y")
""" Generic variable for the player's observations. """

RP = TypeVar("RP")
""" Generic variable for the Personal Reward. """

RJ = TypeVar("RJ")
""" Generic variable for the Joint Reward. """

SR = TypeVar("SR")
""" Generic variable for the type of resources. """

W = TypeVar("W", int, float, Fraction, D)
""" Generic variable for the weight of rewards. """

JointState = Mapping[PlayerName, X]
""" A joint state: the state for each player. """

PlayerOptions = Mapping[PlayerName, FrozenSet[U]]
""" List of options for each player """

JointPureActions = Mapping[PlayerName, U]
""" A pure action for each player"""

JointMixedActions = Mapping[PlayerName, Poss[U]]
""" A mixed strategy for each player"""


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Combined(Generic[RJ, RP]):
    """ A combined cost: personal cost, plus an optional joint cost."""

    personal: RP
    """ The personal total cost. """

    joint: Optional[RJ] = None
    """ The joint final cost. Can be none if we finished without colliding."""

    # Monoid for sum of Combined outcome
    def __add__(self, other: "Combined[RP,RJ]"):
        if type(other) == type(self):
            if other.joint is None:
                return replace(self, personal=self.personal+other.personal, joint=self.joint)
            elif self.joint is None:
                return replace(self, personal=self.personal+other.personal, joint=other.joint)
            else:
                return replace(self, personal=self.personal+other.personal, joint=self.joint+other.joint)
        else:
            raise NotImplementedError

    __radd__ = __add__

    # Weight multiplication (e.g. used to compute expected value)
    def __mul__(self, weight: W):
        # weighting costs, e.g. according to a probability
        if self.joint is not None:
            return replace(self, personal=self.personal*weight, joint=self.joint*weight)
        else:
            return replace(self, personal=self.personal*weight)

    __rmul__ = __mul__


UncertainCombined = Poss[Combined[RP, RJ]]
""" A distribution of combined costs """


class Dynamics(Generic[X, U, SR], ABC):
    """ Dynamics of an agent."""

    @abstractmethod
    def all_actions(self) -> FrozenSet[U]:
        """ Returns all actions possible (not all are available at each state). """

    @abstractmethod
    def successors(self, x: X, dt: D) -> Mapping[U, Poss[X]]:
        """ For each state, returns a dictionary U -> Possible Xs """

    @abstractmethod
    def get_shared_resources(self, x: X) -> FrozenSet[SR]:
        """ Returns the "shared resources" for each state. For example,
            the set of spatio-temporal cells occupied by the agent. """


class Observations(Generic[X, Y], ABC):
    """ Observations of an agent."""

    @abstractmethod
    def all_observations(self) -> FrozenSet[Y]:
        """ Returns all possible observations. """

    @abstractmethod
    def get_observations(self, me: X, others: JointState) -> Poss[Y]:
        """ For each state, get all possible observations """


class PersonalRewardStructure(Generic[X, U, RP], ABC):
    """ The personal reward structure for the agent."""

    @abstractmethod
    def personal_reward_incremental(self, x: X, u: U, dt: D) -> RP:
        """What cost are paid at state X when choosing action u"""

    @abstractmethod
    def personal_reward_reduce(self, r1: RP, r2: RP) -> RP:
        """ How to accumulate reward (sum, monoid operation) """

    @abstractmethod
    def personal_reward_identity(self) -> RP:
        """ The identity for the monoid"""

    @abstractmethod
    def is_personal_final_state(self, x: X) -> bool:
        """ True if this is a final state from the perspective of the agent. """

    @abstractmethod
    def personal_final_reward(self, x: X) -> RP:
        """ Final reward """


P = TypeVar("P")
# fixme not sure why typechecker does not like this new type
MonadicPreferenceBuilder = NewType("MonadicPreferenceBuilder", Callable[[Preference[P]], Preference[Poss[P]]])


@dataclass
class GamePlayer(Generic[X, U, Y, RP, RJ, SR]):
    """ Information about one player. """

    initial: Poss[X]
    """ Initial states """

    dynamics: Dynamics[X, U, SR]
    """ Player dynamics """

    observations: Observations[X, Y]
    """ Player observations """

    personal_reward_structure: PersonalRewardStructure[X, U, RP]
    """ Personal reward information """

    preferences: Preference[Combined[RJ, RP]]
    """ Its preferences about the combined joint/personal rewards. """

    monadic_preference_builder: MonadicPreferenceBuilder
    """ How to evaluate preferences over monadic outcomes. """


@dataclass
class JointRewardStructure(Generic[X, U, RJ], ABC):
    """
        The joint reward structure. This describes when the game ends
        due to "collisions".
    """

    @abstractmethod
    def is_joint_final_state(self, xs: JointState) -> FrozenSet[PlayerName]:
        """ For which players is this a final state? """

    @abstractmethod
    def joint_reward(self, xs: JointState) -> Mapping[PlayerName, RJ]:
        """ The joint reward for the agents. Only available for a final state. """


class GameVisualization(Generic[X, U, Y, RP, RJ], ABC):
    """ An artist that can draw the game. """

    @abstractmethod
    def plot_arena(self, pylab, ax):
        """ Context manager. Plots the arena. """
        pass

    @abstractmethod
    def plot_player(
            self, player_name: PlayerName, state: X, commands: Optional[U], opacity: float = 1.0,
    ):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        pass

    @abstractmethod
    def hint_graph_node_pos(self, state: X) -> Tuple[float, float]:
        pass


@dataclass
class Game(Generic[X, U, Y, RP, RJ, SR]):
    """ Definition of the game """

    ps: PossibilityMonad
    """ Possibility monad to use for this game."""

    players: Mapping[PlayerName, GamePlayer[X, U, Y, RP, RJ, SR]]
    """ The players in this game. """

    joint_reward: JointRewardStructure[X, U, RJ]
    """ The joint reward structure. """

    game_visualization: GameVisualization[X, U, Y, RP, RJ]
    """ The artist to draw this game. """


class AgentBelief(Generic[X, U], ABC):
    """
        This agent's policy is a function of its own state
        and the product of the beliefs about the state of the other agents.
    """

    @abstractmethod
    def get_commands(self, state_self: X, state_others: Poss[JointState]) -> Poss[U]:
        """ Given a state and a belief about the others, produce a distribution of actions to take. """


def check_joint_state(js: JointState, **kwargs):
    """ Checks js is a :any:`JointState`."""
    if not GameConstants.checks:
        return

    check_isinstance(js, frozendict)
    for n, x in js.items():
        check_isinstance(n, str, **kwargs)
        if x is None:
            raise ZValueError(js=js, **kwargs)


def check_player_options(a: PlayerOptions, **kwargs):
    """ Checks consistency of a PlayerOptions variable."""
    if not GameConstants.checks:
        return

    check_isinstance(a, frozendict, **kwargs)
    for k, v in a.items():
        check_isinstance(k, str)
        check_isinstance(v, frozenset)


def check_joint_pure_actions(a: JointPureActions, **kwargs):
    """ Checks consistency of a JointPureActions variable."""
    if not GameConstants.checks:
        return

    check_isinstance(a, frozendict, **kwargs)
    if len(a) == 0:
        raise ZValueError("empty actions", a=a)
    for k, v in a.items():
        assert isinstance(k, str), k
        if isinstance(v, Poss):
            msg = "I thought this would be pure actions, found Poss inside"
            raise ZValueError(msg, k=k, v=v, **kwargs)


def check_joint_mixed_actions2(a: JointMixedActions, **kwargs):
    """ Checks consistency of a JointMixedActions variable."""
    if not GameConstants.checks:
        return
    check_isinstance(a, frozendict, **kwargs)

    for k, v in a.items():
        check_isinstance(k, str)  # player name
        check_isinstance(v, Poss, **kwargs)
        for _ in v.support():
            if isinstance(_, Poss):
                raise ZValueError(_=_, **kwargs)


@dataclass
class BayesianGamePlayer(GamePlayer):
    types: List
