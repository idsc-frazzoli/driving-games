from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, Generic, Mapping, NewType

from frozendict import frozendict
from networkx import MultiDiGraph
from zuper_commons.types import check_isinstance

from preferences import Preference
from .game_def import (check_player_options, PlayerOptions, Pr,
                       check_joint_mixed_actions2,
                       check_joint_pure_actions,
                       check_joint_state,
                       check_set_outcomes,
                       Game,
                       JointMixedActions2,
                       JointPureActions,
                       JointState,
                       PlayerName,
                       RJ,
                       RP,
                       SetOfOutcomes,
                       U,
                       X,
                       Y,
                       )
from .possibilities import Poss
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
STRATEGY_MIX = StrategyForMultipleNash("strategy-mix")
STRATEGY_SECURITY = StrategyForMultipleNash("strategy-security")
STRATEGY_BAIL = StrategyForMultipleNash("strategy-bail")


@dataclass
class SolverParams:
    dt: D
    strategy_multiple_nash: StrategyForMultipleNash


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[Pr, X, U, Y, RP, RJ]):
    states: JointState
    moves: PlayerOptions
    outcomes: "Mapping[JointPureActions, GameNode[Pr, X, U, Y, RP, RJ]]"

    is_final: Mapping[PlayerName, RP]
    incremental: Mapping[PlayerName, Mapping[U, RP]]

    joint_final_rewards: Mapping[PlayerName, RJ]

    def __post_init__(self):
        check_joint_state(self.states)
        check_player_options(self.moves)
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
class GamePlayerPreprocessed(Generic[Pr, X, U, Y, RP, RJ]):
    player_graph: MultiDiGraph
    alone_tree: Mapping[X, GameNode[Pr, X, U, Y, RP, RJ]]


@dataclass
class GamePreprocessed(Generic[Pr, X, U, Y, RP, RJ]):
    game: Game[Pr, X, U, Y, RP, RJ]
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[Pr, X, U, Y, RP, RJ]]
    game_graph: MultiDiGraph
    solver_params: SolverParams


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[U, RP, RJ]):
    mixed_actions: JointMixedActions2
    game_value: SetOfOutcomes

    def __post_init__(self):
        check_joint_mixed_actions2(self.mixed_actions)
        check_set_outcomes(self.game_value)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[Pr, X, U, Y, RP, RJ]):
    gn: GameNode[Pr, X, U, Y, RP, RJ]
    solved: "Mapping[JointPureActions, SolvedGameNode[Pr, X, U, Y, RP, RJ]]"

    va: ValueAndActions[U, RP, RJ]

    # actions: Mapping[PlayerName, ASet[U]]
    # game_value: ASet[Outcome[RP, RJ]]

    def __post_init__(self):
        check_isinstance(self.va, ValueAndActions, me=self)
        check_isinstance(self.solved, frozendict, _=self)
        for _ in self.solved:
            check_joint_pure_actions(_)


@dataclass
class SolvingContext(Generic[Pr, X, U, Y, RP, RJ]):
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]
    cache: Dict[GameNode, SolvedGameNode]
    outcome_set_preferences: Mapping[PlayerName, Preference[SetOfOutcomes]]


@dataclass
class IterationContext:
    gp: GamePreprocessed[Pr, X, U, Y, RP, RJ]
    cache: dict
    depth: int


@dataclass
class GameSolution(Generic[Pr, X, U, Y, RP, RJ]):
    gn: GameNode[Pr, X, U, Y, RP, RJ]
    gn_solved: SolvedGameNode[Pr, X, U, Y, RP, RJ]

    policies: Mapping[PlayerName, Mapping[X, Mapping[Poss[JointState, Pr], Poss[U, Pr]]]]

    def __post_init__(self):
        if False:
            for player_name, player_policy in self.policies.items():

                check_isinstance(player_policy, frozendict)
                for own_state, state_policy in player_policy.items():
                    check_isinstance(state_policy, frozendict)
                    for istate, us in state_policy.items():
                        check_isinstance(us, frozenset)


@dataclass
class SolutionsPlayer(Generic[Pr, X, U, Y, RP, RJ]):
    alone_solutions: Mapping[X, GameSolution]


@dataclass
class Solutions(Generic[Pr, X, U, Y, RP, RJ]):
    solutions_players: Mapping[PlayerName, SolutionsPlayer]
    game_solution: GameSolution[Pr, X, U, Y, RP, RJ]
    game_tree: GameNode[Pr, X, U, Y, RP, RJ]

    sims: Mapping[str, Simulation]
