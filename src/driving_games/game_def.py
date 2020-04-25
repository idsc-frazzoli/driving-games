from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal as D
from typing import AbstractSet, Generic, Mapping, NewType, Optional, Tuple, TypeVar

from networkx import DiGraph

X = TypeVar("X")
U = TypeVar("U")
Y = TypeVar("Y")
RP = TypeVar("RP")
RJ = TypeVar("RJ")

PlayerName = NewType("PlayerName", str)


class Dynamics(Generic[X, U], ABC):
    # @abstractmethod
    # def all_states(self) -> AbstractSet[X]:
    #     """ Returns all possible states """

    @abstractmethod
    def all_actions(self) -> AbstractSet[U]:
        """ Returns all actions possible (not all are available at each state). """

    @abstractmethod
    def successors(self, x: X, dt: D) -> Mapping[U, AbstractSet[X]]:
        """ For each state, returns a dictionary U -> Possible Xs """


class Observations(Generic[X, Y], ABC):
    @abstractmethod
    def all_observations(self) -> AbstractSet[Y]:
        """ Returns all possible observations. """

    @abstractmethod
    def get_observations(self, me: X, others: Mapping[PlayerName, X]) -> AbstractSet[Y]:
        """ For each state, get all possible observations """


class PersonalRewardStructure(Generic[X, U, RP], ABC):
    @abstractmethod
    def personal_reward_incremental(self, x: X, u: U) -> RP:
        """What cost are paid at state X when choosing action u"""

    @abstractmethod
    def personal_reward_reduce(self, r1: RP, r2: RP) -> RP:
        """ How to accumulate reward (sum) """

    @abstractmethod
    def is_personal_final_state(self, x: X) -> bool:
        """ True if this is a final state from the perspective of the agent. """


P = TypeVar("P")


class Poset(Generic[P]):
    @abstractmethod
    def leq(self, a: P, b: P) -> bool:
        """ <= for the poset """


@dataclass
class GamePlayer(Generic[X, U, Y, RP, RJ]):
    # Initial states
    initial: AbstractSet[X]
    # The dynamics
    dynamics: Dynamics[X, U]
    # The observations
    observations: Observations[X, Y]
    # The reward
    personal_reward_structure: PersonalRewardStructure[X, U, RP]
    # The preferences
    preferences: Poset[Tuple[Optional[RJ], RP]]


@dataclass
class JointRewardStructure(Generic[X, U, RJ], ABC):
    @abstractmethod
    def is_joint_final_state(self, xs: Mapping[PlayerName, X]) -> AbstractSet[PlayerName]:
        """ For which players is this a final state? """

    @abstractmethod
    def joint_reward(self, xs: Mapping[PlayerName, X]) -> Mapping[PlayerName, RJ]:
        """ The joint reward for the agents. Only available for a final state. """


@dataclass
class Game(Generic[X, U, Y, RP, RJ]):
    """ The players """

    players: Mapping[PlayerName, GamePlayer[X, U, Y, RP, RJ]]
    """ The joint reward structure """
    joint_reward: JointRewardStructure[X, U, RJ]


@dataclass
class GamePlayerPreprocessed:
    player_graph: DiGraph


@dataclass
class GamePreprocessed:
    game: Game
    dt: D
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed]
    game_graph: DiGraph
