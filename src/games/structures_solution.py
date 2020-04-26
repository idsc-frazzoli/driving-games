from dataclasses import dataclass
from decimal import Decimal as D
from typing import Generic, Mapping

from frozendict import frozendict
from zuper_commons.types import check_isinstance

from preferences import Preference
from .game_def import ASet, GamePreprocessed, PlayerName, RJ, RP, U, X, Y


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[X, U, Y, RP, RJ]):
    states: Mapping[PlayerName, X]
    moves: Mapping[PlayerName, ASet[U]]
    outcomes: "Mapping[Mapping[PlayerName, ASet[U]], GameNode[X, U, Y, RP, RJ]]"

    is_final: Mapping[PlayerName, RP]
    incremental: Mapping[PlayerName, Mapping[U, RP]]

    joint_final_rewards: Mapping[PlayerName, RJ]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class Outcome(Generic[RP, RJ]):
    private: Mapping[PlayerName, RP]
    joint: Mapping[PlayerName, RJ]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[U, RP, RJ]):
    actions: Mapping[PlayerName, ASet[U]]
    game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.actions, frozendict, me=self)
        for k, v in self.actions.items():
            check_isinstance(v, frozenset, me=self, k=k)
        check_isinstance(self.game_value, frozenset, me=self)
        for o in self.game_value:
            check_isinstance(o, Outcome, me=self)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[X, U, Y, RP, RJ]):
    gn: GameNode[X, U, Y, RP, RJ]
    solved: "Mapping[Mapping[PlayerName, ASet[U]], SolvedGameNode[X, U, Y, RP, RJ]]"

    va: ValueAndActions[U, RP, RJ]

    # actions: Mapping[PlayerName, ASet[U]]
    # game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.va, ValueAndActions, me=self)


@dataclass
class SolvingContext(Generic[X, U, Y, RP, RJ]):
    gp: GamePreprocessed[X, U, Y, RP, RJ]
    cache: dict
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
