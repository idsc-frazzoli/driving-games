from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from typing import Callable, FrozenSet, Generic, Mapping, NewType, Optional, Tuple, TypeVar

from dg_commons import PlayerName, U, X, RP, RJ, Y, Timestamp, DgSampledSequence
from possibilities import Poss, PossibilityMonad
from preferences import Preference

__all__ = [
    "Dynamics",
    "SR",
    "P",
    "PlayerOptions",
    "Observations",
    "JointState",
    "JointTransition",
    "JointPureActions",
    "JointMixedActions",
    "JointRewardStructure",
    "PersonalRewardStructure",
    "Combined",
    "UncertainCombined",
    "MonadicPreferenceBuilder",
    "UncertaintyParams",
    "Game",
    "GameSpec",
    "GamePlayer",
    "AgentBelief",
    "GameVisualization",
]

SR = TypeVar("SR")
""" Generic variable for the type of resources. """

JointState = Mapping[PlayerName, X]
""" A joint state: the state for each player. """

JointTransition = Mapping[PlayerName, DgSampledSequence[X]]
""" A joint Transition: a sequence of players states. """

PlayerOptions = Mapping[PlayerName, FrozenSet[U]]
""" List of options for each player (available actions)"""

JointPureActions = Mapping[PlayerName, U]
""" A pure action for each player"""

JointMixedActions = Mapping[PlayerName, Poss[U]]
""" A mixed strategy for each player"""


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Combined(Generic[RJ, RP]):
    """A combined cost: personal cost, plus an optional joint cost."""

    personal: RP
    """ The personal total cost. """

    joint: Optional[RJ] = None
    """ The joint final cost. Can be none if we finished without colliding."""

    # Monoid for sum of Combined outcome
    def __add__(self, other: "Combined[RP,RJ]"):
        if isinstance(other, Combined):
            if other.joint is None:
                return replace(self, personal=self.personal + other.personal, joint=self.joint)
            elif self.joint is None:
                return replace(self, personal=self.personal + other.personal, joint=other.joint)
            else:
                return replace(self, personal=self.personal + other.personal, joint=self.joint + other.joint)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    # Weight multiplication (e.g. used to compute expected value)
    def __mul__(self, weight: Fraction):
        # weighting costs, e.g. according to a probability
        if self.joint is not None:
            return replace(self, personal=self.personal * weight, joint=self.joint * weight)
        else:
            return replace(self, personal=self.personal * weight)

    __rmul__ = __mul__


UncertainCombined = Poss[Combined[RP, RJ]]
""" A distribution of combined costs """


class Dynamics(Generic[X, U, SR], ABC):
    """Dynamics of an agent."""

    @abstractmethod
    def all_actions(self) -> FrozenSet[U]:
        """Returns all actions possible (not all are available at each state)."""

    @abstractmethod
    def successors(self, x: X, dt: D) -> Mapping[U, Poss[X]]:
        """For each state, returns a dictionary U -> Possible Xs"""

    @abstractmethod
    def get_shared_resources(self, x: X, dt: D) -> FrozenSet[SR]:
        """Returns the "shared resources" for each state. For example,
        the set of spatio-temporal cells occupied by the agent."""


class Observations(Generic[X, Y], ABC):
    """Observations of an agent."""

    @abstractmethod
    def all_observations(self) -> FrozenSet[Y]:
        """Returns all possible observations."""

    @abstractmethod
    def get_observations(self, me: X, others: JointState) -> Poss[Y]:
        """For each state, get all possible observations"""


class PersonalRewardStructure(Generic[X, U, RP], ABC):
    """The personal reward structure for the agent."""

    @abstractmethod
    def personal_reward_incremental(self, x: X, u: U, dt: D) -> RP:
        """What cost are paid at state X when choosing action u"""

    @abstractmethod
    def personal_final_reward(self, x: X) -> RP:
        """Final reward"""

    @abstractmethod
    def personal_reward_reduce(self, r1: RP, r2: RP) -> RP:
        """How to accumulate reward (sum, monoid operation)"""

    @abstractmethod
    def personal_reward_identity(self) -> RP:
        """The identity for the monoid"""

    @abstractmethod
    def is_personal_final_state(self, x: X) -> bool:
        """True if this is a final state from the perspective of the agent."""


P = TypeVar("P")

MonadicPreferenceBuilder = NewType("MonadicPreferenceBuilder", Callable[[Preference[P]], Preference[Poss[P]]])


@dataclass(frozen=True, unsafe_hash=True)
class UncertaintyParams:
    poss_monad: PossibilityMonad
    mpref_builder: MonadicPreferenceBuilder


@dataclass
class GamePlayer(Generic[X, U, Y, RP, RJ, SR]):
    """Information about one player."""

    initial: Poss[X]
    """ Initial states """

    dynamics: Dynamics[X, U, SR]
    """ Player dynamics """

    observations: Optional[Observations[X, Y]]
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
    The joint reward structure. This describes when the game ends due to "collisions" and
    the incremental costs due to joint transitions (e.g. minimum safety distance).
    """

    @abstractmethod
    def joint_reward_reduce(self, r1: RJ, r2: RJ) -> RJ:
        """How to accumulate reward (sum, monoid operation)"""

    @abstractmethod
    def joint_reward_identity(self) -> RJ:
        """The identity for the monoid"""

    @abstractmethod
    def joint_reward_incremental(self, txs: JointTransition) -> Mapping[PlayerName, RJ]:
        """The joint incremental reward for the agents."""

    @abstractmethod
    def is_joint_final_transition(self, txs: JointTransition) -> FrozenSet[PlayerName]:
        """For which players this is a final joint transition?"""

    @abstractmethod
    def joint_final_reward(self, txs: JointState) -> Mapping[PlayerName, RJ]:
        """The joint reward for the agents. Only available for a final state."""

    @abstractmethod
    def is_joint_final_states(self, xs: JointState) -> FrozenSet[PlayerName]:
        """the game can end on transition but also on the final state for the players."""


class GameVisualization(Generic[X, U, Y, RP, RJ], ABC):
    """An artist that can draw the game."""

    @abstractmethod
    def plot_arena(self, pylab, ax):
        """Context manager. Plots the arena."""
        pass

    @abstractmethod
    def plot_player(
        self,
        player_name: PlayerName,
        state: X,
        commands: Optional[U],
        t: Timestamp,
        opacity: float = 1.0,
    ):
        """Draw the player at a certain state doing certain commands (if given)"""
        pass

    @abstractmethod
    def hint_graph_node_pos(self, state: X) -> Tuple[float, float]:
        pass


@dataclass
class Game(Generic[X, U, Y, RP, RJ, SR]):
    """Definition of the game"""

    ps: PossibilityMonad
    """ Possibility monad to use for this game."""

    players: Mapping[PlayerName, GamePlayer[X, U, Y, RP, RJ, SR]]
    """ The players in this game. """

    joint_reward: JointRewardStructure[X, U, RJ]
    """ The joint reward structure. """

    game_visualization: GameVisualization[X, U, Y, RP, RJ]
    """ The artist to draw this game. """


@dataclass
class GameSpec:
    desc: str
    game: Game


class AgentBelief(Generic[X, U], ABC):
    """
    This agent's policy is a function of its own state
    and the product of the beliefs about the state of the other agents.
    """

    @abstractmethod
    def get_commands(self, state_self: X, state_others: Poss[JointState]) -> Poss[U]:
        """Given a state and a belief about the others, produce a distribution of actions to take."""
