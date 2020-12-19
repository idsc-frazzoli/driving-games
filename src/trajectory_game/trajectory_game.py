from abc import abstractmethod
from functools import partial
from typing import Dict, Set, FrozenSet, Mapping
from time import perf_counter

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE
from games.utils import iterate_dict_combinations
from possibilities import Poss
from preferences import (
    Preference,
)

from .structures import VehicleState
from .paths import Trajectory
from .world import World
from .metrics_def import PlayerOutcome, TrajGameOutcome
from .static_game import (
    StaticGame,
    StaticGamePlayer,
    StaticSolvingContext,
    StaticSolvedGameNode,
    ActionSetGenerator, StaticSolverParams,
)

__all__ = [
    "JointPureTraj",
    "TrajectoryGamePlayer",
    "TrajectoryGame",
    "SolvedTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "compute_solving_context",
]

# JointTrajSet = Mapping[PlayerName, FrozenSet[Trajectory]]  # fixme this seems a bit confusing..
JointPureTraj = Mapping[PlayerName, Trajectory]
JointMixedTraj = Mapping[PlayerName, Poss[Trajectory]]


class TrajectoryGenerator(ActionSetGenerator[VehicleState, Trajectory, World]):
    @abstractmethod
    def get_actions_set(self, state: Poss[VehicleState], world: World) -> FrozenSet[Trajectory]:
        """ Generate all possible actions for a given state and world. """


class TrajectoryGamePlayer(StaticGamePlayer[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


class TrajectoryGame(StaticGame[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


class SolvedTrajectoryGameNode(StaticSolvedGameNode[Trajectory, PlayerOutcome]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]


def compute_solving_context(sgame: StaticGame) -> StaticSolvingContext:
    """
    Preprocess the game -> Compute all possible actions and outcomes for each combination
    """
    ps = sgame.ps

    # Generate the trajectories for each player (i.e. get the available actions)
    available_traj: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, game_player in sgame.game_players.items():
        # In the future can be extended to uncertain initial state
        states = game_player.state.support()
        assert len(states) == 1, states
        available_traj[player_name] = game_player.actions_generator.get_action_set(
            state=next(iter(states)), world=sgame.world
        )

    # Compute the distribution of outcomes for each joint action
    tic = perf_counter()
    outcomes: Dict[JointPureTraj, Poss[TrajGameOutcome]] = {}
    get_outcomes = partial(sgame.get_outcomes, world=sgame.world)
    for joint_traj in set(iterate_dict_combinations(available_traj)):
        outcomes[joint_traj] = ps.build(ps.unit(joint_traj), f=get_outcomes)

    toc = perf_counter() - tic
    print(f"Outcomes evaluation time = {toc} s")

    # Similar to get_outcome_preferences_for_players, use SetPreference1 for Poss
    pref: Mapping[PlayerName, Preference[PlayerOutcome]] = {
        name: player.preference for name, player in sgame.game_players.items()
    }

    context = StaticSolvingContext(
        player_actions=available_traj,
        game_outcomes=outcomes,
        outcome_pref=pref,  # todo I fear here it's missing the monadic preferences but it is fine for now
        solver_params=StaticSolverParams(
            admissible_strategies=PURE_STRATEGIES,
            strategy_multiple_nash=BAIL_MNE  # this is not used for now
        )
    )
    return context


# def solve_game(context: StaticSolvingContext) ->\
#         Tuple[SolvedTrajectoryGame, SolvedTrajectoryGame,
#               SolvedTrajectoryGame, SolvedTrajectoryGame]:

#     indiff_nash: SolvedTrajectoryGame = set()
#     incomp_nash: SolvedTrajectoryGame = set()
#     weak_nash: SolvedTrajectoryGame = set()
#     strong_nash: SolvedTrajectoryGame = set()

#     def get_action_options(joint_act: JointTrajProfile, pname: PlayerName) -> JointTrajSet:
#         """Returns all possible actions for the player, with other player actions frozen
#            Current player action is not included"""

#         def get_actions(name: PlayerName) -> FrozenSet[Trajectory]:
#             if name == pname:
#                 p_actions: Set[Trajectory] = set(actions[name])
#                 p_actions.remove(joint_act[pname])
#                 return frozenset(p_actions)
#             return frozenset({joint_act[name]})

#         ret: JointTrajSet = {_: get_actions(_) for _ in players}
#         return ret

#     def get_solved_game_node(act: JointTrajProfile, out: TrajectoryGameOutcome) -> \
#             SolvedTrajectoryGameNode:
#         return SolvedTrajectoryGameNode(actions=act, outcomes=out)

#     # TODO[SIR]: Remove dominated options first or just brute force through?
#     actions = context.player_actions
#     players = actions.keys()

#     tic = perf_counter()
#     # For each possible action combination, check if it is a nash eq
#     for joint_actions in get_joint_traj(actions):

#         # For each player, compare the current outcome to their alternatives
#         # Classify into types of nash eq. based on the outputs
#         outcome: TrajectoryGameOutcome = context.game_outcomes[joint_actions]
#         results: Set[ComparisonOutcome] = set()

#         for player in players:
#             action_alt: JointTrajSet = get_action_options(joint_act=joint_actions, pname=player)
#             player_outcome: PlayerOutcome = outcome[player]
#             for joint_act_alt in get_joint_traj(action_alt):
#                 player_outcome_alt: PlayerOutcome = \
#                     context.game_outcomes[joint_act_alt][player]
#                 comp_outcome: ComparisonOutcome = \
#                     context.outcome_pref[player].compare(player_outcome, player_outcome_alt)
#                 results.add(comp_outcome)

#                 # If second option is preferred, current point is not a nash eq.
#                 if comp_outcome == SECOND_PREFERRED:
#                     break

#             # If second option is preferred for any player, current point is not a nash eq.
#             if SECOND_PREFERRED in results:
#                 break

#         # If second option is preferred for any player, current point is not a nash eq.
#         if SECOND_PREFERRED in results:
#             continue

#         solved_node: SolvedTrajectoryGameNode = \
#             get_solved_game_node(act=joint_actions, out=outcome)
#         if results == {FIRST_PREFERRED}:
#             strong_nash.add(solved_node)
#             continue
#         if INDIFFERENT in results and INCOMPARABLE in results:
#             weak_nash.add(solved_node)
#             indiff_nash.add(solved_node)
#             incomp_nash.add(solved_node)
#         elif INDIFFERENT in results:
#             indiff_nash.add(solved_node)
#             weak_nash.add(solved_node)
#         elif INCOMPARABLE in results:
#             incomp_nash.add(solved_node)
#             weak_nash.add(solved_node)

#     toc = perf_counter() - tic
#     print('Nash equilibrium computation time = {} s'.format(toc))

#     return indiff_nash, incomp_nash, weak_nash, strong_nash
