from copy import deepcopy
from random import choice
from time import perf_counter
from typing import Mapping, Dict, FrozenSet, Set, Tuple, Optional
from frozendict import frozendict

from games.utils import iterate_dict_combinations
from preferences import ComparisonOutcome, SECOND_PREFERRED, INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED

from games import PlayerName
from .game_def import StaticSolvingContext, DynamicSolvingContext
from .trajectory_game import JointPureTraj, SolvedTrajectoryGameNode, SolvedTrajectoryGame, \
    SubgameSolutions
from .paths import Trajectory, TrajectoryGraph
from .metrics_def import TrajGameOutcome, PlayerOutcome

JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]
EqOutcome = Tuple[Optional[JointPureTraj], Optional[TrajGameOutcome], bool, bool, bool, bool]
NotEq: EqOutcome = None, None, False, False, False, False


def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
    return SolvedTrajectoryGameNode(actions=act, outcomes=out)


def callback_eq(tuple_out: EqOutcome, eq: Dict[str, SolvedTrajectoryGame]):
    joint_act, outcome, strong, incomp, indiff, weak = tuple_out

    solved_node: SolvedTrajectoryGameNode = \
        get_solved_game_node(act=joint_act, out=outcome) if weak else None
    eq["indiff"].add(solved_node) if indiff else None
    eq["incomp"].add(solved_node) if incomp else None
    eq["weak"].add(solved_node) if weak else None
    eq["strong"].add(solved_node) if strong else None


def init_eq_dict() -> Dict[str, SolvedTrajectoryGame]:
    ret: Dict[str, SolvedTrajectoryGame] = {
        "indiff": set(), "incomp": set(),
        "weak": set(), "strong": set(),
    }
    return ret


def check_dominated(joint_actions: JointPureTraj,
                    done: Mapping[PlayerName, Set[JointPureTraj]]) -> bool:
    for dominated in done.values():
        if joint_actions in dominated:
            return True
    return False


def get_best_responses(joint_actions: JointPureTraj, context: StaticSolvingContext,
                       player: PlayerName, done_p: Set[JointPureTraj]) \
        -> Tuple[Set[ComparisonOutcome], Set[Trajectory]]:
    """
    Calculates the best responses for the current player
    Returns best responses and comparison outcomes
    """
    actions = context.player_actions
    players = joint_actions.keys()
    final = SubgameSolutions()

    # All alternate actions (from available set) for the current player
    def get_action_options(joint_act: JointPureTraj, p_actions: Set[Trajectory]) -> JointTrajSet:
        """Returns all possible actions for the player, with other player actions frozen
        Current player action is not included"""

        def get_actions(name: PlayerName) -> FrozenSet[Trajectory]:
            if name == player:
                return frozenset(p_actions)
            return frozenset({joint_act[name]})

        action_options: JointTrajSet = {_: get_actions(_) for _ in players}
        return action_options

    # Antichain of outcomes
    def get_antichain(actions_joint: JointPureTraj) -> Set[Trajectory]:
        antichain_all = final.get_trajectories(joint_traj=actions_joint)
        # TODO[SIR]: Use full antichain
        return {next(iter(antichain_all))[player]}

    all_actions: Set[Trajectory] = set(actions[player])
    all_actions.remove(joint_actions[player])

    # Save antichain of actions
    best = get_antichain(actions_joint=joint_actions)
    joint_best: Dict[PlayerName, Trajectory] = {k: v + None for k, v in joint_actions.items()}
    results: Set[ComparisonOutcome] = set()
    check = False
    action_alt: JointTrajSet = get_action_options(joint_act=joint_actions,
                                                  p_actions=all_actions)
    for joint_act_alt in iterate_dict_combinations(action_alt):
        antichain_alt = get_antichain(actions_joint=joint_act_alt)
        joint_alt: Dict[PlayerName, Trajectory] = {k: v + None for k, v in joint_actions.items()}

        # TODO[SIR]: Antichain comparison
        for alt_action in antichain_alt:
            joint_alt[player] = alt_action + None
            joint_alt_f = frozendict(joint_alt)
            if joint_alt_f in done_p:
                continue
            results_alt: Set[ComparisonOutcome] = set()
            alt_outcome: PlayerOutcome = context.game_outcomes(joint_alt_f)[player]

            for best_action in list(best):
                if best_action in antichain_alt:
                    continue
                check = True
                joint_best[player] = best_action
                joint_best_f = frozendict(joint_best)
                best_outcome: PlayerOutcome = context.game_outcomes(joint_best_f)[player]
                comp_outcome: ComparisonOutcome = \
                    context.outcome_pref[player].compare(best_outcome, alt_outcome)
                results_alt.add(comp_outcome)
                if best_action == joint_actions[player]:
                    results.add(comp_outcome)

                # If one of best is preferred, alternate is not a best response
                if comp_outcome == FIRST_PREFERRED:
                    done_p.add(joint_alt_f)
                    break

                # If second option is preferred, current best action is not a best response
                if comp_outcome == SECOND_PREFERRED:
                    best.remove(best_action)
                    done_p.add(joint_best_f)

            if FIRST_PREFERRED not in results_alt:
                best.add(alt_action)

    if not check:
        results.add(FIRST_PREFERRED)
    return results, best


def equilibrium_check(joint_actions: JointPureTraj, context: StaticSolvingContext,
                      done: Dict[PlayerName, Set[JointPureTraj]]) -> EqOutcome:
    """
    For each player, check if current action is best response
    Classify into types of nash eq. based on the outputs
    """
    # TODO[SIR]: Extend to mixed outcomes and strategies

    if isinstance(joint_actions, dict):
        joint_actions = frozendict(joint_actions)
    if check_dominated(joint_actions=joint_actions, done=done):
        return NotEq

    # Compute best responses for each player, check if joint_actions is dominated
    results: Set[ComparisonOutcome] = set()
    for player in joint_actions.keys():
        results_player, _ = get_best_responses(joint_actions=joint_actions, context=context,
                                               player=player, done_p=done[player])
        # If second option is preferred for any player, current point is not a nash eq.
        if SECOND_PREFERRED in results_player:
            return NotEq
        results |= results_player

    outcome: TrajGameOutcome = context.game_outcomes(joint_actions)
    strong = results == {FIRST_PREFERRED}
    indiff = INDIFFERENT in results
    incomp = INCOMPARABLE in results
    weak = True  # All equilibria are atleast weak
    return joint_actions, outcome, strong, incomp, indiff, weak


class Solution:
    # Cache can be reused between levels
    dominated: Dict[PlayerName, Set[JointPureTraj]] = None

    def solve_game(self, context: StaticSolvingContext, cache_dom: bool = False) \
            -> Mapping[str, SolvedTrajectoryGame]:
        eq_dict = init_eq_dict()
        dom_prev = deepcopy(self.dominated) if not cache_dom else {}
        tic = perf_counter()
        # For each possible action combination, check if it is a nash eq
        if self.dominated is None:
            self.dominated = {_: set() for _ in context.player_actions.keys()}
        for joint_act in set(iterate_dict_combinations(context.player_actions)):
            out = equilibrium_check(joint_actions=joint_act, context=context,
                                    done=self.dominated)
            callback_eq(tuple_out=out, eq=eq_dict)
        toc = perf_counter() - tic
        print(f"Nash equilibrium computation time = {toc:.2f} s")
        if not cache_dom:
            self.dominated = dom_prev
        return eq_dict

    def reset(self):
        self.dominated: Dict[PlayerName, Set[JointPureTraj]] = None


def solve_static_game(context: StaticSolvingContext) \
        -> Mapping[str, SolvedTrajectoryGame]:
    sol = Solution()
    eq_dict = sol.solve_game(context=context)
    static_eq: Dict[str, SolvedTrajectoryGame] = {}

    for eq_type, node_set in eq_dict.items():
        static_set = {SolvedTrajectoryGameNode(actions=node.actions, outcomes=node.outcomes)
                      for node in node_set}
        static_eq[eq_type] = static_set

    return static_eq


def iterative_best_response(context: StaticSolvingContext, n_runs: int) \
        -> Mapping[str, SolvedTrajectoryGame]:
    eq_dict = init_eq_dict()
    INIT_BEST = True

    # Solve single player game for each player to get initial guess
    all_actions: Mapping[PlayerName, FrozenSet[Trajectory]] = context.player_actions
    init_guess: Dict[PlayerName, Set[Trajectory]] = {}
    done_p: Set[JointPureTraj] = set()
    tic = perf_counter()
    if INIT_BEST:
        print("Initialising strategies using best personal strategies.\nBest Actions:")
        for player, actions in all_actions.items():
            done_p.clear()
            joint_actions = frozendict({player: next(iter(actions))})
            _, best_actions = get_best_responses(joint_actions=joint_actions, context=context,
                                                 player=player, done_p=done_p)
            print(f"\tPlayer: {player} = {len(best_actions)}")
            init_guess[player] = best_actions
        toc = perf_counter() - tic
        print(f"Individual best computation time = {toc:.2f} s")
    else:
        print("Initialising strategies at random")
        init_guess = {player: actions for player, actions in all_actions.items()}

    done: Dict[PlayerName, Set[JointPureTraj]] = \
        {_: set() for _ in context.player_actions.keys()}
    for i in range(n_runs):
        players_rem: Set[PlayerName] = set(init_guess.keys())
        joint_best: Dict[PlayerName, Trajectory] = {}
        for player, actions in init_guess.items():
            joint_best[player] = choice(list(actions))

        # Select a player at random and change action to one from best response set
        # Continue till joint action is part of best for all players
        while len(players_rem) > 0:
            player = choice(list(players_rem))
            players_rem.remove(player)
            _, best_actions = get_best_responses(joint_actions=joint_best, context=context,
                                                 player=player, done_p=done[player])
            if joint_best[player] not in best_actions:
                players_rem = set(init_guess.keys())
                players_rem.remove(player)
            joint_best[player] = choice(list(best_actions))

        out = equilibrium_check(joint_actions=joint_best, context=context, done=done)
        callback_eq(tuple_out=out, eq=eq_dict)

    toc = perf_counter() - tic
    print(f"Best response equilibrium computation time = {toc:.2f} s")

    return eq_dict


def solve_subgame(context: DynamicSolvingContext):
    all_actions: Mapping[PlayerName, TrajectoryGraph] = context.player_actions
    actions: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player, graph in all_actions.items():
        actions[player] = graph.get_all_trajectories(source=graph.origin)

    static_context = StaticSolvingContext(player_actions=actions,
                                          game_outcomes=context.game_outcomes,
                                          outcome_pref=context.outcome_pref,
                                          solver_params=context.solver_params)
    sol = Solution()
    sol.solve_game(context=static_context)
