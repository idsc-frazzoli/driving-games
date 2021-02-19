from time import perf_counter
from typing import Mapping, Dict, FrozenSet, Set

from games.utils import iterate_dict_combinations
from possibilities import Poss
from preferences import ComparisonOutcome, SECOND_PREFERRED, INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED

from games import PlayerName
from .static_game import StaticSolvingContext
from .trajectory_game import JointPureTraj, SolvedTrajectoryGame, SolvedTrajectoryGameNode
from .paths import Trajectory
from .metrics_def import TrajGameOutcome, PlayerOutcome

JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]


def solve_game(
    context: StaticSolvingContext,
) -> Mapping[str, SolvedTrajectoryGame]:
    indiff_nash: SolvedTrajectoryGame = set()
    incomp_nash: SolvedTrajectoryGame = set()
    weak_nash: SolvedTrajectoryGame = set()
    strong_nash: SolvedTrajectoryGame = set()

    def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
        return SolvedTrajectoryGameNode(actions=act, outcomes=out)

    def callback(tuple_out):
        joint_act, outcome, indiff, incomp, weak, strong = tuple_out
        solved_node: SolvedTrajectoryGameNode = get_solved_game_node(act=joint_act, out=outcome)
        indiff_nash.add(solved_node) if indiff else None
        incomp_nash.add(solved_node) if incomp else None
        weak_nash.add(solved_node) if weak else None
        strong_nash.add(solved_node) if strong else None

    tic = perf_counter()
    # For each possible action combination, check if it is a nash eq
    for val in set(iterate_dict_combinations(context.player_actions)):
        out = equilibria_check(joint_actions=val, context=context)
        callback(out)

    toc = perf_counter() - tic
    print(f"Nash equilibrium computation time = {toc:.2f} s")

    ret: Dict[str, SolvedTrajectoryGame] = {
        "indiff_nash": indiff_nash,
        "incomp_nash": incomp_nash,
        "weak_nash": weak_nash,
        "strong_nash": strong_nash,
    }
    return ret


def equilibria_check(joint_actions, context: StaticSolvingContext):
    # TODO[SIR]: Extend to mixed outcomes and strategies
    actions = context.player_actions
    players = actions.keys()

    def get_action_options(joint_act: JointPureTraj, pname: PlayerName) -> JointTrajSet:
        """Returns all possible actions for the player, with other player actions frozen
        Current player action is not included"""

        def get_actions(name: PlayerName) -> FrozenSet[Trajectory]:
            if name == pname:
                p_actions: Set[Trajectory] = set(actions[name])
                p_actions.remove(joint_act[pname])
                return frozenset(p_actions)
            return frozenset({joint_act[name]})

        action_options: JointTrajSet = {_: get_actions(_) for _ in players}
        return action_options

    # For each player, compare the current outcome to their alternatives
    # Classify into types of nash eq. based on the outputs
    outcome_poss: Poss[TrajGameOutcome] = context.game_outcomes[joint_actions]
    assert len(outcome_poss.support()) == 1, outcome_poss.support()
    outcome: TrajGameOutcome = next(iter(outcome_poss.support()))
    results: Set[ComparisonOutcome] = set()
    for player in players:
        action_alt: JointTrajSet = get_action_options(joint_act=joint_actions, pname=player)
        player_outcome: PlayerOutcome = outcome[player]
        for joint_act_alt in set(iterate_dict_combinations(action_alt)):
            alt_outcome = context.game_outcomes[joint_act_alt].support()
            assert len(alt_outcome) == 1, alt_outcome
            player_outcome_alt: PlayerOutcome = next(iter(alt_outcome))[player]
            comp_outcome: ComparisonOutcome = context.outcome_pref[player].compare(
                player_outcome, player_outcome_alt
            )
            results.add(comp_outcome)

            # If second option is preferred, current point is not a nash eq.
            if comp_outcome == SECOND_PREFERRED:
                break

        # If second option is preferred for any player, current point is not a nash eq.
        if SECOND_PREFERRED in results:
            break

    # If second option is preferred for any player, current point is not a nash eq.
    if SECOND_PREFERRED in results:
        return joint_actions, outcome, False, False, False, False

    strong = results == {FIRST_PREFERRED}
    weak = INDIFFERENT in results or INCOMPARABLE in results
    indiff = INDIFFERENT in results
    incomp = INCOMPARABLE in results
    return joint_actions, outcome, indiff, incomp, weak, strong
