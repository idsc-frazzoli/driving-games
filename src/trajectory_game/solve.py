from time import perf_counter

from possibilities import Poss
from preferences import ComparisonOutcome, SECOND_PREFERRED, INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED

from games import PlayerName
from trajectory_game import JointPureTraj, SolvedTrajectoryGame, Trajectory, Set, SolvedTrajectoryGameNode, \
    TrajGameOutcome, PlayerOutcome, FrozenSet, StaticSolvingContext, Tuple


def solve_game(
    context: StaticSolvingContext,
) -> Tuple[SolvedTrajectoryGame, SolvedTrajectoryGame, SolvedTrajectoryGame, SolvedTrajectoryGame]:
    indiff_nash: SolvedTrajectoryGame = set()
    incomp_nash: SolvedTrajectoryGame = set()
    weak_nash: SolvedTrajectoryGame = set()
    strong_nash: SolvedTrajectoryGame = set()

    def get_solved_game_node(act: JointPureTraj, out: TrajGameOutcome) -> SolvedTrajectoryGameNode:
        return SolvedTrajectoryGameNode(actions=act, outcomes=TrajGameOutcome)

    # TODO[SIR]: Remove dominated options first or just brute force through?
    actions = context.player_actions
    players = actions.keys()

    tic = perf_counter()
    # For each possible action combination, check if it is a nash eq
    for joint_actions in context.player_actions:

        # For each player, compare the current outcome to their alternatives
        # Classify into types of nash eq. based on the outputs
        outcome: Poss[TrajGameOutcome] = context.game_outcomes[joint_actions]
        results: Set[ComparisonOutcome] = set()
        # todo check the similarities with _solve_equilibria
        for player in players:
            action_alt: JointTrajSet = get_action_options(joint_act=joint_actions, pname=player)
            player_outcome: PlayerOutcome = outcome[player]
            for joint_act_alt in get_joint_traj(action_alt):
                player_outcome_alt: PlayerOutcome = context.game_outcomes[joint_act_alt][player]
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

    return indiff_nash, incomp_nash, weak_nash, strong_nash
