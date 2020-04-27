from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal as D
from typing import (
    Callable,
    FrozenSet as ASet,
    Generic,
    Mapping,
    NewType,
    Optional,
    TypeVar,
)

from networkx import MultiDiGraph

from preferences import Preference

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
    def all_actions(self) -> ASet[U]:
        """ Returns all actions possible (not all are available at each state). """

    @abstractmethod
    def successors(self, x: X, dt: D) -> Mapping[U, ASet[X]]:
        """ For each state, returns a dictionary U -> Possible Xs """


class Observations(Generic[X, Y], ABC):
    @abstractmethod
    def all_observations(self) -> ASet[Y]:
        """ Returns all possible observations. """

    @abstractmethod
    def get_observations(self, me: X, others: Mapping[PlayerName, X]) -> ASet[Y]:
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


P = TypeVar("P")


@dataclass
class GamePlayer(Generic[X, U, Y, RP, RJ]):
    # Initial states
    initial: ASet[X]
    # The dynamics
    dynamics: Dynamics[X, U]
    # The observations
    observations: Observations[X, Y]
    # The reward
    personal_reward_structure: PersonalRewardStructure[X, U, RP]
    # The preferences
    preferences: Preference[Combined[RJ, RP]]
    # How to aggregate preferences for sets
    set_preference_aggregator: Callable[[Preference[P]], Preference[ASet[P]]]


@dataclass
class JointRewardStructure(Generic[X, U, RJ], ABC):
    @abstractmethod
    def is_joint_final_state(self, xs: Mapping[PlayerName, X]) -> ASet[PlayerName]:
        """ For which players is this a final state? """

    @abstractmethod
    def joint_reward(self, xs: Mapping[PlayerName, X]) -> Mapping[PlayerName, RJ]:
        """ The joint reward for the agents. Only available for a final state. """



class GameVisualization(Generic[X, U, Y, RP, RJ], ABC):

    @abstractmethod
    def plot_arena(self, pylab, ax):
        pass

    @abstractmethod
    def plot_player(self, player_name: PlayerName, state: X, commands: Optional[U],
                    opacity: float = 1.0):
        """ Draw the player at a certain state doing certain commands (if givne)"""
        pass


@dataclass
class Game(Generic[X, U, Y, RP, RJ]):
    """ The players """
    players: Mapping[PlayerName, GamePlayer[X, U, Y, RP, RJ]]
    """ The joint reward structure """
    joint_reward: JointRewardStructure[X, U, RJ]

    game_visualization: GameVisualization[X, U, Y, RP, RJ]


@dataclass
class GamePlayerPreprocessed(Generic[X, U, Y, RP, RJ]):
    player_graph: MultiDiGraph


@dataclass
class GamePreprocessed(Generic[X, U, Y, RP, RJ]):
    game: Game[X, U, Y, RP, RJ]
    dt: D
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ]]
    game_graph: MultiDiGraph


class AgentBelief(Generic[X, U], ABC):
    """ This agent's policy is a function of its own state
        and the product of the beliefs about the state of the other agents.
    """

    @abstractmethod
    def get_commands(self, state_self: X, state_others: Mapping[PlayerName, ASet[X]]) -> ASet[U]:
        ...


JointState = Mapping[PlayerName, Optional[X]]
JointAction = Mapping[PlayerName, Optional[U]]
