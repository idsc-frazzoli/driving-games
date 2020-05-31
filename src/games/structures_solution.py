from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D
from typing import AbstractSet, Dict, FrozenSet as FSet, Generic, Mapping, NewType, Set

from frozendict import frozendict
from networkx import MultiDiGraph

from possibilities import check_poss, Poss
from preferences import Preference
from zuper_commons.types import check_isinstance, ZValueError
from . import GameConstants
from .game_def import (
    check_joint_mixed_actions2,
    check_joint_pure_actions,
    check_joint_state,
    check_player_options,
    Combined,
    Game,
    JointMixedActions,
    JointPureActions,
    JointState,
    PlayerName,
    PlayerOptions,
    RJ,
    RP,
    SR,
    U,
    UncertainCombined,
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

from .utils import fkeyfilter, iterate_dict_combinations

StrategyForMultipleNash = NewType("StrategyForMultipleNash", str)
STRATEGY_MIX = StrategyForMultipleNash("mix")
STRATEGY_SECURITY = StrategyForMultipleNash("security")
STRATEGY_BAIL = StrategyForMultipleNash("bail")


@dataclass
class SolverParams:
    dt: D
    strategy_multiple_nash: StrategyForMultipleNash
    use_factorization: bool


@dataclass(frozen=True, unsafe_hash=True, order=True)
class GameNode(Generic[X, U, Y, RP, RJ, SR]):
    """ The game node """

    states: JointState
    """ The state for each player."""

    moves: PlayerOptions
    """ 
        The possible moves for each player. Note that each player must have at least one move.
        In case there is only one move, then there is nothing for them to choose to do.
    """

    outcomes: Mapping[JointPureActions, Poss[Mapping[PlayerName, JointState]]]
    """ 
        The outcomes. Fixed an action for each player (a JointPureAction), we have
        a distribution of outcomes. Each outcome is a map that tells us which 
        player goes to which joint state. This is a generalization beyond the usual
        formalization of games that allows us to send different players to different games.
        
        For example, suppose that the next state is = {A:x, B:y} and that in those states
        the play can be decoupled. Then we would send A to the game {A:x} and B to the game {B:y}.
    """

    is_final: Mapping[PlayerName, RP]
    """ Final cost for the players that terminate here."""

    incremental: Mapping[PlayerName, Mapping[U, RP]]
    """ Incremental cost according to action taken. """

    joint_final_rewards: Mapping[PlayerName, RJ]
    """ For the players that terminate here due to "collision", their final rewards. """

    resources: Mapping[PlayerName, FSet[SR]]
    """ Resources used by each player """

    __print_order__ = [
        "states",
        "moves",
        "outcomes",
        "is_final",
        "incremental",
        "joint_final_rewards",
    ]  # only print
    # these attributes

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        check_joint_state(self.states, GameNode=self)
        check_player_options(self.moves, GameNode=self)
        check_isinstance(self.outcomes, frozendict, GameNode=self)
        for pure_actions, pr_game_node in self.outcomes.items():
            check_joint_pure_actions(pure_actions)
            # check_isinstance(pr_game_node, dict)
            check_poss(pr_game_node, frozendict)
            for x in pr_game_node.support():
                check_isinstance(x, frozendict)
                for k, js in x.items():
                    check_isinstance(k, str)
                    check_joint_state(js)

        check_isinstance(self.is_final, frozendict, _=self)
        check_isinstance(self.incremental, frozendict, _=self)
        check_isinstance(self.joint_final_rewards, frozendict, _=self)

        # check_isinstance(joint_actions, frozendict, _=self)
        for player_name, player_moves in self.moves.items():
            if player_moves == {None}:
                raise ZValueError(_self=self)

        # check that the actions are available
        for jpa in self.outcomes:
            for player_name, action in jpa.items():
                if player_name not in self.moves:
                    msg = f"The player {player_name!r} does not have any moves, so how can it choose?"
                    raise ZValueError(msg, action=action, GameNode=self)
                if action not in self.moves[player_name]:
                    msg = f"The action is not available to the player."
                    raise ZValueError(msg, player_name=player_name, action=action, GameNode=self)
        # check that if a player is not final then it has at least 1 move
        all_players = set(self.states)
        final_players = set(self.is_final) | set(self.joint_final_rewards)
        continuing_players = all_players - final_players
        for player_name in continuing_players:
            if not player_name in self.moves:
                msg = f"Player {player_name!r} is continuing but does not have any move."
                raise ZValueError(msg, GameNode=self)

        # check that we have in outcomes all combinations of actions
        all_combinations = set(iterate_dict_combinations(self.moves))
        if all_combinations == {frozendict()}:
            all_combinations = set()
        if all_combinations != set(self.outcomes):
            msg = "There is a mismatch between the actions and the outcomes."
            raise ZValueError(
                msg, all_combinations=all_combinations, pure_actions=set(self.outcomes), GameNode=self,
            )

        # check that for each action we have a cost
        for player_name, player_moves in self.moves.items():
            moves_with_cost = set(self.incremental[player_name])
            if player_moves != moves_with_cost:
                msg = "Invalid match between moves and costs."
                raise ZValueError(
                    msg,
                    player_name=player_name,
                    player_moves=player_moves,
                    moves_with_cost=moves_with_cost,
                    GameNode=self,
                )

        self.check_players_in_outcome()

    def check_players_in_outcome(self) -> None:
        """ We want to make sure that each player transitions in a game in which he is present. """
        jpa: JointPureActions
        consequences: Poss[Mapping[PlayerName, JointState]]
        for jpa, consequences in self.outcomes.items():
            for new_games in consequences.support():
                for player_name, next_state in new_games.items():
                    if not player_name in next_state:
                        msg = f"The player {player_name!r} is transitioning to a state without it. "
                        raise ZValueError(msg, GameNode=self)

                    if player_name in self.is_final:
                        msg = (
                            f"The player {player_name!r} is transitioning to a state but it is marked as "
                            f"personal final."
                        )
                        raise ZValueError(msg, GameNode=self)
                    if player_name in self.joint_final_rewards:
                        msg = f"The player {player_name!r} is transitioning to a state but it is marked as joint final."
                        raise ZValueError(msg, GameNode=self)


def states_mentioned(game_node: GameNode) -> FSet[JointState]:
    # outcomes: Mapping[JointPureActions, Poss[Mapping[PlayerName, JointState]]]
    res = set()
    for _, out in game_node.outcomes.items():
        for player_to_js in out.support():
            for player_name, js in player_to_js.items():
                res.add(js)
    return frozenset(res)


@dataclass
class AccessibilityInfo(Generic[X]):
    state2times: Dict[JointState, Set[D]]
    time2states: Dict[D, Set[JointState]]


@dataclass
class GameGraph(Generic[X, U, Y, RP, RJ, SR]):
    initials: AbstractSet[JointState]
    state2node: Mapping[JointState, GameNode[X, U, Y, RP, RJ, SR]]
    ti: AccessibilityInfo[X]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        # Check that there are all states mentioned
        mentioned = defaultdict(set)
        for joint_state, game_node in self.state2node.items():
            for m in states_mentioned(game_node):
                mentioned[m].add(joint_state)
        missing = set(mentioned) - set(self.state2node)
        if missing:
            m = fkeyfilter(missing.__contains__, mentioned)
            msg = "There are states mentioned but missing"
            raise ZValueError(msg, missing_and_mention=m)


@dataclass
class GamePlayerPreprocessed(Generic[X, U, Y, RP, RJ, SR]):
    player_graph: MultiDiGraph
    # alone_tree: Mapping[X, GameNode[X, U, Y, RP, RJ, SR]]
    game_graph: GameGraph[X, U, Y, RP, RJ, SR]
    gs: "GameSolution[X, U, Y, RP, RJ, SR]"


@dataclass
class GameFactorization(Generic[X]):
    partitions: Mapping[FSet[FSet[PlayerName]], FSet[JointState]]
    ipartitions: Mapping[JointState, FSet[FSet[PlayerName]]]


@dataclass
class GamePreprocessed(Generic[X, U, Y, RP, RJ, SR]):
    game: Game[X, U, Y, RP, RJ, SR]
    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]]
    game_graph: MultiDiGraph
    solver_params: SolverParams
    game_factorization: GameFactorization[X]


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions2(Generic[U, RP, RJ]):
    mixed_actions: JointMixedActions
    game_value: Mapping[PlayerName, UncertainCombined]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return
        check_isinstance(self.game_value, frozendict, ValueAndActions=self)
        for _ in self.game_value.values():
            check_poss(_, Combined, ValueAndActions=self)
        check_joint_mixed_actions2(self.mixed_actions, ValueAndActions=self)
        # check_set_outcomes(self.game_value, ValueAndActions=self)
        # check_isinstance(self.game_values, frozendict, ValueAndActions=self)
        # check_set_outcomes(self.game_value, ValueAndActions=self)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class UsedResources(Generic[X, U, Y, RP, RJ, SR]):
    # Used resources at each time.
    # D = 0 means now. +1 means next step, etc.
    used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]]


M = Mapping


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[X, U, Y, RP, RJ, SR]):
    states: JointState
    solved: M[JointPureActions, Poss[M[PlayerName, JointState]]]
    # solved: "Mapping[JointPureActions, Poss[SolvedGameNode[X, U, Y, RP, RJ, SR]]]"

    va: ValueAndActions2[U, RP, RJ]

    ur: UsedResources[X, U, Y, RP, RJ, SR]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        check_isinstance(self.va, ValueAndActions2, SolvedGameNode=self)
        check_isinstance(self.solved, frozendict, SolvedGameNode=self)
        for jpa, then in self.solved.items():
            check_joint_pure_actions(jpa)
            check_poss(then, SolvedGameNode=self)

        players = list(self.states)

        for p in players:
            if p not in self.va.game_value:
                msg = f"There is no player {p!r} appearing in the game value"
                raise ZValueError(msg, SolvedGameNode=self)


@dataclass
class SolvingContext(Generic[X, U, Y, RP, RJ, SR]):
    game: Game[X, U, Y, RP, RJ, SR]
    # gp: GamePreprocessed[X, U, Y, RP, RJ, SR]
    outcome_set_preferences: Mapping[PlayerName, Preference[UncertainCombined]]
    cache: Dict[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]
    processing: Set[JointState]
    gg: GameGraph[X, U, Y, RP, RJ, SR]
    solver_params: SolverParams


@dataclass
class GameSolution(Generic[X, U, Y, RP, RJ, SR]):
    initials: AbstractSet[JointState]
    states_to_solution: Dict[JointState, SolvedGameNode]
    policies: Mapping[PlayerName, Mapping[X, Mapping[Poss[JointState], Poss[U]]]]

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        for player_name, player_policy in self.policies.items():

            check_isinstance(player_policy, frozendict)
            for own_state, state_policy in player_policy.items():
                check_isinstance(state_policy, frozendict)
                for istate, us in state_policy.items():
                    check_poss(us)


@dataclass
class SolutionsPlayer(Generic[X, U, Y, RP, RJ, SR]):
    alone_solutions: Mapping[X, GameSolution[X, U, Y, RP, RJ, SR]]


@dataclass
class Solutions(Generic[X, U, Y, RP, RJ, SR]):
    solutions_players: Mapping[PlayerName, SolutionsPlayer[X, U, Y, RP, RJ, SR]]
    game_solution: GameSolution[X, U, Y, RP, RJ, SR]
    game_tree: GameNode[X, U, Y, RP, RJ, SR]

    sims: Mapping[str, Simulation]
