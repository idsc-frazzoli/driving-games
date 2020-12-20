from time import perf_counter
from typing import Mapping, Dict, FrozenSet, Set, Tuple

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

    def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
        return SolvedTrajectoryGameNode(actions=act, outcomes=out)

    # TODO[SIR]: Remove dominated options first or just brute force through?
    actions = context.player_actions
    players = actions.keys()

    # todo check the similarities with solve_equilibria -->
    #  Outcomes are different, what else?
    tic = perf_counter()

    # TODO[SIR]: Extend to mixed outcomes and strategies
    # For each possible action combination, check if it is a nash eq
    for joint_actions in set(iterate_dict_combinations(context.player_actions)):

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
            continue

        solved_node: SolvedTrajectoryGameNode = get_solved_game_node(act=joint_actions, out=outcome)
        if results == {FIRST_PREFERRED}:
            strong_nash.add(solved_node)
            continue
        if INDIFFERENT in results and INCOMPARABLE in results:
            weak_nash.add(solved_node)
            indiff_nash.add(solved_node)
            incomp_nash.add(solved_node)
        elif INDIFFERENT in results:
            indiff_nash.add(solved_node)
            weak_nash.add(solved_node)
        elif INCOMPARABLE in results:
            incomp_nash.add(solved_node)
            weak_nash.add(solved_node)

    toc = perf_counter() - tic
    print("Nash equilibrium computation time = {} s".format(toc))

    ret: Dict[str, SolvedTrajectoryGame] = {
        "indiff_nash": indiff_nash,
        "incomp_nash": incomp_nash,
        "weak_nash": weak_nash,
        "strong_nash": strong_nash,
    }
    return ret
