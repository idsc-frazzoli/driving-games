from random import choice
from time import perf_counter
from typing import Mapping, Dict, FrozenSet, Set, Tuple
from frozendict import frozendict

from games.utils import iterate_dict_combinations
from preferences import ComparisonOutcome, SECOND_PREFERRED, INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED

from games import PlayerName
from .static_game import StaticSolvingContext
from .trajectory_game import JointPureTraj, SolvedTrajectoryGame, SolvedTrajectoryGameNode
from .paths import Trajectory
from .metrics_def import TrajGameOutcome, PlayerOutcome

JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]
EqOutcome = Tuple[JointPureTraj, TrajGameOutcome, bool, bool, bool, bool]


def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
    return SolvedTrajectoryGameNode(actions=act, outcomes=out)


def callback_eq(tuple_out: EqOutcome, eq: Dict[str, SolvedTrajectoryGame]):
    joint_act, outcome, strong, incomp, indiff, weak = tuple_out

    solved_node: SolvedTrajectoryGameNode = get_solved_game_node(act=joint_act, out=outcome) \
        if weak or strong else None
    eq["indiff_nash"].add(solved_node) if indiff else None
    eq["incomp_nash"].add(solved_node) if incomp else None
    eq["weak_nash"].add(solved_node) if weak else None
    eq["strong_nash"].add(solved_node) if strong else None


def init_eq_dict() -> Dict[str, SolvedTrajectoryGame]:
    indiff_nash: SolvedTrajectoryGame = set()
    incomp_nash: SolvedTrajectoryGame = set()
    weak_nash: SolvedTrajectoryGame = set()
    strong_nash: SolvedTrajectoryGame = set()
    ret: Dict[str, SolvedTrajectoryGame] = {
        "indiff_nash": indiff_nash,
        "incomp_nash": incomp_nash,
        "weak_nash": weak_nash,
        "strong_nash": strong_nash,
    }
    return ret


def check_best_response(joint_actions: JointPureTraj, context: StaticSolvingContext,
                        player: PlayerName, return_best: bool) \
        -> Tuple[Set[ComparisonOutcome], JointPureTraj]:
    actions = context.player_actions
    players = joint_actions.keys()

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

    # Track all actions available - need to compare all incomp with best response at the end
    available_actions: Set[Trajectory] = set(actions[player])
    available_actions.remove(joint_actions[player])

    results: Set[ComparisonOutcome] = set()
    converged = False
    while not converged and len(available_actions) > 0:
        action_alt: JointTrajSet = get_action_options(joint_act=joint_actions,
                                                      p_actions=available_actions)
        player_outcome: PlayerOutcome = context.game_outcomes(joint_actions)[player]
        results = set()
        converged = True
        for joint_act_alt in set(iterate_dict_combinations(action_alt)):
            player_action = joint_act_alt[player]
            player_outcome_alt: PlayerOutcome = context.game_outcomes(joint_act_alt)[player]
            comp_outcome: ComparisonOutcome = \
                context.outcome_pref[player].compare(player_outcome, player_outcome_alt)
            if comp_outcome != INDIFFERENT:
                results.add(comp_outcome)

            # Keep track of only incomparable actions, others don't need to be checked again
            if comp_outcome != INCOMPARABLE:
                available_actions.remove(player_action)

            # If second option is preferred, current action is not a best response
            if comp_outcome == SECOND_PREFERRED:
                if return_best:
                    joint_actions = joint_act_alt
                    player_outcome = player_outcome_alt
                    results = set()
                    converged = False
                else:
                    return {SECOND_PREFERRED}, joint_actions

    return results, joint_actions


def equilibrium_check(joint_actions: JointPureTraj, context: StaticSolvingContext) -> EqOutcome:
    # TODO[SIR]: Extend to mixed outcomes and strategies

    # For each player, check if current action is best response
    # Classify into types of nash eq. based on the outputs
    results: Set[ComparisonOutcome] = set()
    for player in joint_actions.keys():
        results_player, _ = check_best_response(joint_actions=joint_actions, context=context,
                                                player=player, return_best=False)
        # If second option is preferred for any player, current point is not a nash eq.
        if SECOND_PREFERRED in results_player:
            return None, None, False, False, False, False
        results |= results_player

    outcome: TrajGameOutcome = context.game_outcomes(joint_actions)
    strong = results == {FIRST_PREFERRED}
    incomp = INCOMPARABLE in results
    indiff = not (strong or incomp)
    weak = not strong
    return joint_actions, outcome, strong, incomp, indiff, weak


def solve_game(context: StaticSolvingContext) -> Mapping[str, SolvedTrajectoryGame]:
    eq_dict = init_eq_dict()

    tic = perf_counter()
    # For each possible action combination, check if it is a nash eq
    for val in set(iterate_dict_combinations(context.player_actions)):
        out = equilibrium_check(joint_actions=val, context=context)
        callback_eq(tuple_out=out, eq=eq_dict)
    toc = perf_counter() - tic
    print(f"Nash equilibrium computation time = {toc:.2f} s")

    return eq_dict


def iterative_best_response(context: StaticSolvingContext, n_runs: int) \
        -> Mapping[str, SolvedTrajectoryGame]:
    eq_dict = init_eq_dict()

    # Solve single player game for each player to get initial guess
    tic = perf_counter()
    all_actions = context.player_actions
    indiv_best: Dict[PlayerName, Trajectory] = {}
    for player, actions in all_actions.items():
        joint_actions = frozendict({player: next(iter(actions))})
        _, best_action = check_best_response(joint_actions=joint_actions, context=context,
                                             player=player, return_best=True)
        indiv_best[player] = best_action[player]
    toc = perf_counter() - tic
    print(f"Individual best computation time = {toc:.2f} s")

    for i in range(n_runs):
        players_rem: Set[PlayerName] = set(indiv_best.keys())
        joint_best: JointPureTraj = frozendict(indiv_best)

        # Select a player at random and change action to best response
        # Continue till joint action is best for all players
        while len(players_rem) > 0:
            player = choice(list(players_rem))
            players_rem.remove(player)
            _, update_best = check_best_response(joint_actions=joint_best, context=context,
                                                 player=player, return_best=True)
            if joint_best != update_best:
                players_rem = set(update_best.keys())
            joint_best = update_best

        out = equilibrium_check(joint_actions=joint_best, context=context)
        callback_eq(tuple_out=out, eq=eq_dict)

    toc = perf_counter() - tic
    print(f"Best response equilibrium computation time = {toc:.2f} s")

    return eq_dict
