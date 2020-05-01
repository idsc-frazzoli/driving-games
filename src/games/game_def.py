from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal as D
from numbers import Number
from typing import (
    Callable,
    # FrozenSet as ASet,
    FrozenSet,
    Generic,
    Mapping,
    NewType,
    Optional,
    Tuple,
    TypeVar,
)

from frozendict import frozendict

from games import GameConstants
from zuper_commons.types import check_isinstance, ZValueError

from possibilities import check_poss, Poss, PossibilityStructure
from preferences import Preference

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
    "JointMixedActions2",
    "JointRewardStructure",
    "Outcome",
    "SetOfOutcomes",
    "PersonalRewardStructure",
    "PlayerName",
    "Combined",
    "Game",
    "GamePlayer",
    "GameVisualization",
]

PlayerName = NewType("PlayerName", str)

X = TypeVar("X")
JointState = Mapping[PlayerName, X]

U = TypeVar("U")
Pr = TypeVar("Pr", bound=Number)  # how to express probabilities

PlayerOptions = Mapping[PlayerName, FrozenSet[U]]
JointPureActions = Mapping[PlayerName, U]
JointMixedActions2 = Mapping[PlayerName, Poss[U, Pr]]

Y = TypeVar("Y")
RP = TypeVar("RP")
RJ = TypeVar("RJ")


@dataclass(frozen=True, unsafe_hash=True, order=True)
class Outcome(Generic[RP, RJ]):
    private: Mapping[PlayerName, RP]
    joint: Mapping[PlayerName, RJ]


SetOfOutcomes = Poss[Outcome[RP, RJ], Pr]


class Dynamics(Generic[Pr, X, U], ABC):
    # @abstractmethod
    # def all_states(self) -> AbstractSet[X]:
    #     """ Returns all possible states """

    @abstractmethod
    def all_actions(self) -> FrozenSet[U]:
        """ Returns all actions possible (not all are available at each state). """

    @abstractmethod
    def successors(self, x: X, dt: D) -> Mapping[U, Poss[X, Pr]]:
        """ For each state, returns a dictionary U -> Possible Xs """


class Observations(Generic[Pr, X, Y], ABC):
    @abstractmethod
    def all_observations(self) -> FrozenSet[Y]:
        """ Returns all possible observations. """

    @abstractmethod
    def get_observations(self, me: X, others: JointState) -> Poss[Y, Pr]:
        """ For each state, get all possible observations """


class PersonalRewardStructure(Generic[X, U, RP], ABC):
    @abstractmethod
    def personal_reward_incremental(self, x: X, u: U, dt: D) -> RP:
        """What cost are paid at state X when choosing action u"""

    @abstractmethod
    def personal_reward_reduce(self, r1: RP, r2: RP) -> RP:
        """ How to accumulate reward (sum) """

    @abstractmethod
    def is_personal_final_state(self, x: X) -> bool:
        """ True if this is a final state from the perspective of the agent. """

    @abstractmethod
    def personal_final_reward(self, x: X) -> RP:
        """ Final reward """


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Combined(Generic[RJ, RP]):
    personal: RP
    joint: Optional[RJ]


#
P = TypeVar("P")


@dataclass
class GamePlayer(Generic[Pr, X, U, Y, RP, RJ]):
    # Initial states
    initial: Poss[X, Pr]
    # The dynamics
    dynamics: Dynamics[Pr, X, U]
    # The observations
    observations: Observations[Pr, X, Y]
    # The reward
    personal_reward_structure: PersonalRewardStructure[X, U, RP]
    # The preferences
    preferences: Preference[Combined[RJ, RP]]
    # How to aggregate preferences for sets
    set_preference_aggregator: Callable[[Preference[P]], Preference[Poss[P, Pr]]]


@dataclass
class JointRewardStructure(Generic[X, U, RJ], ABC):
    @abstractmethod
    def is_joint_final_state(self, xs: JointState) -> FrozenSet[PlayerName]:
        """ For which players is this a final state? """

    @abstractmethod
    def joint_reward(self, xs: JointState) -> Mapping[PlayerName, RJ]:
        """ The joint reward for the agents. Only available for a final state. """


class GameVisualization(Generic[Pr, X, U, Y, RP, RJ], ABC):
    @abstractmethod
    def plot_arena(self, pylab, ax):
        """ Context manager """
        pass

    @abstractmethod
    def plot_player(self, player_name: PlayerName, state: X, commands: Optional[U], opacity: float = 1.0):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        pass

    @abstractmethod
    def hint_graph_node_pos(self, state: X) -> Tuple[float, float]:
        pass


@dataclass
class Game(Generic[Pr, X, U, Y, RP, RJ]):
    """ The players """

    ps: PossibilityStructure[Pr]

    players: Mapping[PlayerName, GamePlayer[Pr, X, U, Y, RP, RJ]]
    """ The joint reward structure """
    joint_reward: JointRewardStructure[X, U, RJ]

    game_visualization: GameVisualization[Pr, X, U, Y, RP, RJ]


class AgentBelief(Generic[Pr, X, U], ABC):
    """ This agent's policy is a function of its own state
        and the product of the beliefs about the state of the other agents.
    """

    @abstractmethod
    def get_commands(self, state_self: X, state_others: Poss[JointState, Pr]) -> Poss[U, Pr]:
        ...


def check_joint_state(js: JointState):
    # from driving_games import VehicleState  # XXX : for debug
    if not GameConstants.checks:
        return

    check_isinstance(js, frozendict)
    for n, x in js.items():
        check_isinstance(n, str)
        # check_isinstance(x, VehicleState)


def check_player_options(a: PlayerOptions):
    if not GameConstants.checks:
        return

    check_isinstance(a, frozendict)
    for k, v in a.items():
        check_isinstance(k, str)
        check_isinstance(v, frozenset)


def check_set_outcomes(a: SetOfOutcomes, **kwargs):
    if not GameConstants.checks:
        return

    check_poss(a, Outcome, **kwargs)


def check_joint_pure_actions(a: JointPureActions, **kwargs):
    if not GameConstants.checks:
        return

    # from driving_games.structures import VehicleActions

    check_isinstance(a, frozendict, **kwargs)
    for k, v in a.items():
        assert isinstance(k, str), k
        if isinstance(v, Poss):
            msg = "I thought this would be pure actions, found Poss inside"
            raise ZValueError(msg, k=k, v=v, **kwargs)
        pass
        # check_isinstance(v, VehicleActions, a=a)


def check_joint_mixed_actions2(a: JointMixedActions2, **kwargs):
    if not GameConstants.checks:
        return
    # from driving_games.structures import VehicleActions
    check_isinstance(a, frozendict, **kwargs)

    for k, v in a.items():
        check_isinstance(k, str)  # player name
        check_isinstance(v, Poss, **kwargs)
        for _ in v.support():
            if isinstance(_, Poss):
                raise ZValueError(_=_, **kwargs)

    # check_isinstance(_, VehicleActions, a=a)
