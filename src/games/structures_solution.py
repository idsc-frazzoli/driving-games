from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic, Mapping

from frozendict import frozendict
from networkx import MultiDiGraph
from zuper_commons.types import check_isinstance

from preferences import Preference
from .game_def import (
    ASet,
    check_joint_state,
    JointMixedActions,
    JointState,
    PlayerName,
    RJ,
    RP,
    U,
    X,
    Y,
    Game,
    JointPureActions,
)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class Outcome(Generic[RP, RJ]):
    private: Mapping[PlayerName, RP]
    joint: Mapping[PlayerName, RJ]


SetOfOutcomes = ASet[Outcome[RP, RJ]]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[X, U, Y, RP, RJ]):
    states: JointState
    moves: JointMixedActions
    outcomes: "Mapping[JointPureActions, GameNode[X, U, Y, RP, RJ]]"

    is_final: Mapping[PlayerName, RP]
    incremental: Mapping[PlayerName, Mapping[U, RP]]

    joint_final_rewards: Mapping[PlayerName, RJ]

    def __post_init__(self):
        check_joint_state(self.states)
        check_joint_mixed_actions(self.moves)
        check_isinstance(self.outcomes, frozendict, _=self)
        for pure_actions, game_node in self.outcomes.items():
            check_joint_pure_actions(pure_actions)
            check_isinstance(game_node, GameNode)

        check_isinstance(self.is_final, frozendict, _=self)
        check_isinstance(self.incremental, frozendict, _=self)
        check_isinstance(self.joint_final_rewards, frozendict, _=self)

        # check_isinstance(joint_actions, frozendict, _=self)
        # for player_name, actions in self.outcomes.items():
        #     if isinstance(actions, (set, frozenset)):
        #         raise ZValueError(_=self)
        # assert type(actions).__name__ in ['VehicleActions'], actions


@dataclass
class GamePlayerPreprocessed(Generic[X, U, Y, RP, RJ]):
    player_graph: MultiDiGraph
    alone_tree: Mapping[X, GameNode[X, U, Y, RP, RJ]]


@dataclass
class GamePreprocessed(Generic[X, U, Y, RP, RJ]):
    game: Game[X, U, Y, RP, RJ]
    dt: D
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ]]
    game_graph: MultiDiGraph


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[U, RP, RJ]):
    mixed_actions: JointMixedActions
    game_value: SetOfOutcomes

    def __post_init__(self):
        check_joint_mixed_actions(self.mixed_actions)
        check_set_outcomes(self.game_value)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[X, U, Y, RP, RJ]):
    gn: GameNode[X, U, Y, RP, RJ]
    solved: "Mapping[JointPureActions, SolvedGameNode[X, U, Y, RP, RJ]]"

    va: ValueAndActions[U, RP, RJ]

    # actions: Mapping[PlayerName, ASet[U]]
    # game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.va, ValueAndActions, me=self)
        check_isinstance(self.solved, frozendict, _=self)
        for _ in self.solved:
            check_joint_pure_actions(_)


@dataclass
class SolvingContext(Generic[X, U, Y, RP, RJ]):
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: Dict[GameNode, SolvedGameNode]
    depth: int
    outcome_set_preferences: Mapping[PlayerName, Preference[ASet[Outcome]]]


@dataclass
class IterationContext:
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: dict
    depth: int


@dataclass
class SolverParams:
    dt: D


def check_set_outcomes(a: SetOfOutcomes):
    check_isinstance(a, frozenset)
    for _ in a:
        check_isinstance(_, Outcome, a=a)


def check_joint_pure_actions(a: JointPureActions):
    from driving_games.structures import VehicleActions

    check_isinstance(a, frozendict)
    for k, v in a.items():
        check_isinstance(v, VehicleActions, a=a)


def check_joint_mixed_actions(a: JointMixedActions):
    from driving_games.structures import VehicleActions

    check_isinstance(a, frozendict)
    for k, v in a.items():
        check_isinstance(v, frozenset, a=a)
        for _ in v:
            check_isinstance(_, VehicleActions, a=a)
