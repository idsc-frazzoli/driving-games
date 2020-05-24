from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, FrozenSet as FSet, Generic, Mapping, NewType, Set

from frozendict import frozendict
from networkx import MultiDiGraph

from possibilities import check_poss, Poss
from preferences import Preference
from zuper_commons.types import check_isinstance
from . import GameConstants
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    check_joint_state,
    check_player_options,
    check_set_outcomes,
    Game,
    JointMixedActions2,
    JointPureActions,
    JointState,
    PlayerName,
    PlayerOptions,
    Pr,
    RJ,
    RP,
    SetOfOutcomes,
    SR,
    U,
    X,
    Y,
)
from .simulate import Simulation

__all__ = [
    "GameNode",
    "GamePreprocessed",
    "GamePlayerPreprocessed",
    "SolvedGameNode",
    "SolverParams",
    "SolvingContext",
    "STRATEGY_BAIL",
    "STRATEGY_SECURITY",
    "STRATEGY_MIX",
    "StrategyForMultipleNash",
]

StrategyForMultipleNash = NewType("StrategyForMultipleNash", str)
STRATEGY_MIX = StrategyForMultipleNash("mix")
STRATEGY_SECURITY = StrategyForMultipleNash("security")
STRATEGY_BAIL = StrategyForMultipleNash("bail")


@dataclass
class SolverParams:
    dt: D
    strategy_multiple_nash: StrategyForMultipleNash


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[Pr, X, U, Y, RP, RJ, SR]):
    states: JointState
    moves: PlayerOptions
    outcomes2: "Mapping[JointPureActions, Poss[GameNode[Pr, X, U, Y, RP, RJ, SR], Pr]]"

    is_final: Mapping[PlayerName, RP]
    incremental: Mapping[PlayerName, Mapping[U, RP]]

    joint_final_rewards: Mapping[PlayerName, RJ]

    resources: Mapping[PlayerName, FSet[SR]]

    def __post_init__(self):
        if not GameConstants.checks:
            return

        check_joint_state(self.states)
        check_player_options(self.moves)
        check_isinstance(self.outcomes2, frozendict, _=self)
        for pure_actions, pr_game_node in self.outcomes2.items():
            check_joint_pure_actions(pure_actions)
            check_poss(pr_game_node, GameNode)

        check_isinstance(self.is_final, frozendict, _=self)
        check_isinstance(self.incremental, frozendict, _=self)
        check_isinstance(self.joint_final_rewards, frozendict, _=self)

        # check_isinstance(joint_actions, frozendict, _=self)
        # for player_name, actions in self.outcomes.items():
        #     if isinstance(actions, (set, frozenset)):
        #         raise ZValueError(_=self)
        # assert type(actions).__name__ in ['VehicleActions'], actions


@dataclass
class GamePlayerPreprocessed(Generic[Pr, X, U, Y, RP, RJ]):
    player_graph: MultiDiGraph
    alone_tree: Mapping[X, GameNode[Pr, X, U, Y, RP, RJ, SR]]


@dataclass
class GameFactorization(Generic[X]):
    partitions: Mapping[FSet[FSet[PlayerName]], FSet[JointState]]
    ipartitions: Mapping[JointState, FSet[FSet[PlayerName]]]

@dataclass
class GamePreprocessed(Generic[Pr, X, U, Y, RP, RJ]):
    game: Game[Pr, X, U, Y, RP, RJ, SR]
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[Pr, X, U, Y, RP, RJ]]
    game_graph: MultiDiGraph
    solver_params: SolverParams
    game_factorization: GameFactorization[X]

@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[Pr, U, RP, RJ]):
    mixed_actions: JointMixedActions2
    game_value: SetOfOutcomes

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        check_joint_mixed_actions2(self.mixed_actions, ValueAndActions=self)
        check_set_outcomes(self.game_value, ValueAndActions=self)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class UsedResources(Generic[Pr, X, U, Y, RP, RJ, SR]):
    # Used resources at each time.
    # D = 0 means now. +1 means next step, etc.
    used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]], Pr]]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[Pr, X, U, Y, RP, RJ, SR]):
    gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
    solved: "Mapping[JointPureActions, Poss[SolvedGameNode[Pr, X, U, Y, RP, RJ, SR], Pr]]"

    va: ValueAndActions[Pr, U, RP, RJ]

    ur: UsedResources[Pr, X, U, Y, RP, RJ, SR]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        check_isinstance(self.va, ValueAndActions, me=self)
        check_isinstance(self.solved, frozendict, _=self)
        for _, then in self.solved.items():
            check_joint_pure_actions(_)
            check_poss(then, SolvedGameNode)


@dataclass
class SolvingContext(Generic[Pr, X, U, Y, RP, RJ]):
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]
    outcome_set_preferences: Mapping[PlayerName, Preference[SetOfOutcomes]]
    cache: Dict[GameNode, SolvedGameNode]
    processing: Set[JointState]


@dataclass
class IterationContext:
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]
    cache: dict
    depth: int


@dataclass
class GameSolution(Generic[Pr, X, U, Y, RP, RJ]):
    gn: GameNode[Pr, X, U, Y, RP, RJ, SR]
    gn_solved: SolvedGameNode[Pr, X, U, Y, RP, RJ, SR]

    states_to_solution: Dict[JointState, SolvedGameNode]

    policies: Mapping[PlayerName, Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]]

    def __post_init__(self):
        if not GameConstants.checks:
            return

        for player_name, player_policy in self.policies.items():

            check_isinstance(player_policy, frozendict)
            for own_state, state_policy in player_policy.items():
                check_isinstance(state_policy, frozendict)
                for istate, us in state_policy.items():
                    check_poss(us)


@dataclass
class SolutionsPlayer(Generic[Pr, X, U, Y, RP, RJ]):
    alone_solutions: Mapping[X, GameSolution]


@dataclass
class Solutions(Generic[Pr, X, U, Y, RP, RJ]):
    solutions_players: Mapping[PlayerName, SolutionsPlayer]
    game_solution: GameSolution[Pr, X, U, Y, RP, RJ]
    game_tree: GameNode[Pr, X, U, Y, RP, RJ, SR]

    sims: Mapping[str, Simulation]
