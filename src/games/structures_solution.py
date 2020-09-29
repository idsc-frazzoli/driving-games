from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D
from typing import AbstractSet, Dict, FrozenSet as FSet, Generic, Mapping, NewType, Set, Mapping as M

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
    PlayerBelief
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
""" How to deal with multiple nash equilibria. """

STRATEGY_MIX = StrategyForMultipleNash("mix")
""" Mix all the states in the multiple nash equilibria. """

STRATEGY_SECURITY = StrategyForMultipleNash("security")
""" Use a securety policy. """

STRATEGY_BAIL = StrategyForMultipleNash("bail")
""" Throw an error. """


@dataclass
class SolverParams:
    """ Parameters for the solver"""

    dt: D
    """ The delta-t when discretizing. """
    strategy_multiple_nash: StrategyForMultipleNash
    """ How to deal with multiple Nash equilibria """
    use_factorization: bool
    """ Whether to use the factorization properties to reduce the game graph."""


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

    incremental: Mapping[PlayerName, Mapping[U, Poss[RP]]]
    """ Incremental cost according to action taken. """

    joint_final_rewards: Mapping[PlayerName, RJ]
    """ For the players that terminate here due to "collision", their final rewards. """

    resources: Mapping[PlayerName, FSet[SR]]
    """ Resources used by each player """

    belief: Mapping[PlayerName, PlayerBelief]
    """ Belief of each player at this game node """

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
    """ Returns the set of state mentioned in a GameNode"""
    res = set()
    for _, out in game_node.outcomes.items():
        for player_to_js in out.support():
            for player_name, js in player_to_js.items():
                res.add(js)
    return frozenset(res)


@dataclass
class AccessibilityInfo(Generic[X]):
    """ The time accessibility info of the states of a game"""

    state2times: Dict[JointState, AbstractSet[D]]
    """ For each state, at what time can it be visited? """

    time2states: Dict[D, AbstractSet[JointState]]
    """ For each time, what states can be visited? """


@dataclass
class GameGraph(Generic[X, U, Y, RP, RJ, SR]):
    """ The game graph."""

    initials: AbstractSet[JointState]
    """ The initial states of the game. """

    state2node: Mapping[JointState, GameNode[X, U, Y, RP, RJ, SR]]
    """ 
        Maps to each joint state a GameNode. Inside a GameNode the next
        states are identified by their JointState only. 
    """

    ti: AccessibilityInfo[X]
    """ The time accessibility info for the states. """

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
    """ Pre-processed data for each game player"""

    player_graph: MultiDiGraph
    """ A NetworkX graph used fo visualization. """

    game_graph: GameGraph[X, U, Y, RP, RJ, SR]
    """ The game graph for the alone game. """

    gs: "GameSolution[X, U, Y, RP, RJ, SR]"
    """ The solution to the alone game."""


@dataclass
class GameFactorization(Generic[X]):
    """ Factorization information for the game"""

    partitions: Mapping[FSet[FSet[PlayerName]], FSet[JointState]]
    """ For each partition of players, what joint states have that partition? """

    ipartitions: Mapping[JointState, FSet[FSet[PlayerName]]]
    """ For each joint state, how can we partition the players? """


@dataclass
class GamePreprocessed(Generic[X, U, Y, RP, RJ, SR]):
    """ A pre-processed game. """

    game: Game[X, U, Y, RP, RJ, SR]
    """ The original game. """

    players_pre: Mapping[PlayerName, GamePlayerPreprocessed[X, U, Y, RP, RJ, SR]]
    """ The pre-processed data for each player"""

    game_graph: MultiDiGraph
    """ A NetworkX graph used only for visualization """

    solver_params: SolverParams
    """ The solver parameters. """

    game_factorization: GameFactorization[X]
    """ The factorization information for the game"""


@dataclass(frozen=True, unsafe_hash=True, order=True)
class ValueAndActions(Generic[U, RP, RJ]):
    """ The solution for a game node. """

    mixed_actions: JointMixedActions
    """ What players choose. """

    game_value: Mapping[PlayerName, UncertainCombined]
    """ What is the value of the game for each player. """

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return
        check_isinstance(self.game_value, frozendict, ValueAndActions=self)
        for _ in self.game_value.values():
            check_poss(_, Combined, ValueAndActions=self)
        check_joint_mixed_actions2(self.mixed_actions, ValueAndActions=self)


@dataclass(frozen=True, unsafe_hash=True, order=True)
class UsedResources(Generic[X, U, Y, RP, RJ, SR]):
    """ The used *future* resources for a particular state. """

    used: Mapping[D, Poss[Mapping[PlayerName, FSet[SR]]]]
    """
        For each delta time (D = 0 means now. +1 means next step, etc.) 
        what states are the agents going to use. 
        For each delta time we have a distribution of spatial resource occupancy.
    """


@dataclass(frozen=True, unsafe_hash=True, order=True)
class SolvedGameNode(Generic[X, U, Y, RP, RJ, SR]):
    """ A solved game node. """

    states: JointState
    """ The joint state for this node. """

    solved: M[JointPureActions, Poss[M[PlayerName, JointState]]]
    """ For each joint action, this is the outcome (where each player goes). """

    va: ValueAndActions[U, RP, RJ]
    """ The strategy profiles and the game values"""

    ur: UsedResources[X, U, Y, RP, RJ, SR]
    """ The future used resources. """

    def __post_init__(self) -> None:
        if not GameConstants.checks:
            return

        check_isinstance(self.va, ValueAndActions, SolvedGameNode=self)
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
    """ Context for the solution of the game"""

    game: Game[X, U, Y, RP, RJ, SR]
    """ The original game. """

    outcome_preferences: Mapping[PlayerName, Preference[UncertainCombined]]
    """ The preferences of each player"""

    cache: Dict[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]
    """ The nodes already solved."""

    processing: Set[JointState]
    """ The nodes currently processing. """

    gg: GameGraph[X, U, Y, RP, RJ, SR]
    """ The game graph. """

    solver_params: SolverParams
    """ The solver parameters. """


@dataclass
class GameSolution(Generic[X, U, Y, RP, RJ, SR]):
    """ Solution of a game. """

    initials: AbstractSet[JointState]
    """ Set of initial states for which we have a solution """

    states_to_solution: Dict[JointState, SolvedGameNode]
    """ The solution of each state. """

    policies: Mapping[PlayerName, Mapping[X, Mapping[Poss[JointState], Poss[U]]]]
    """ The policies resulting from this solution. For each player, for each state,
        a map from belief on others to distribution of actions. """

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
