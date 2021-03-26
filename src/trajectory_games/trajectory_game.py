from functools import partial
from typing import Dict, Set, FrozenSet, Mapping
from time import perf_counter

from frozendict import frozendict
from networkx import MultiDiGraph

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE
from games.utils import iterate_dict_combinations
from preferences import Preference

from .structures import VehicleState, VehicleGeometry
from .paths import Transition, Trajectory
from .trajectory_world import TrajectoryWorld
from .metrics_def import PlayerOutcome
from .static_game import StaticGame, StaticGamePlayer, StaticSolvingContext,\
    StaticSolvedGameNode, StaticSolverParams

__all__ = [
    "JointPureTraj",
    "JointTrans",
    "StaticTrajectoryGamePlayer",
    "StaticTrajectoryGame",
    "SolvedStaticTrajectoryGameNode",
    "SolvedStaticTrajectoryGame",
    "preprocess_full_game",
    "preprocess_player",
]

JointPureTraj = Mapping[PlayerName, Trajectory]
JointTrans = Mapping[PlayerName, Transition]


class StaticTrajectoryGamePlayer(StaticGamePlayer[VehicleState, Trajectory,
                                                  TrajectoryWorld, PlayerOutcome,
                                                  VehicleGeometry]):
    pass


class StaticTrajectoryGame(StaticGame[VehicleState, Trajectory,
                                      TrajectoryWorld, PlayerOutcome,
                                      VehicleGeometry]):
    pass


class SolvedStaticTrajectoryGameNode(StaticSolvedGameNode[Trajectory, PlayerOutcome]):
    pass


SolvedStaticTrajectoryGame = Set[SolvedStaticTrajectoryGameNode]


def compute_outcomes(iterable, sgame: StaticGame):
    key, joint_traj_in = iterable
    get_outcomes = partial(sgame.get_outcomes, world=sgame.world)
    ps = sgame.ps
    return key, ps.build(ps.unit(joint_traj_in), f=get_outcomes)


def compute_actions(sgame: StaticGame) -> Mapping[PlayerName, FrozenSet[Trajectory]]:
    """ Generate the trajectories for each player (i.e. get the available actions) """
    available_traj: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, game_player in sgame.game_players.items():
        # In the future can be extended to uncertain initial state
        states = game_player.state.support()
        assert len(states) == 1, states
        game_player.graph = MultiDiGraph()
        available_traj[player_name] = game_player.actions_generator.get_action_set(
            state=next(iter(states)), world=sgame.world, player=player_name
        )
    return available_traj


def preprocess_player(sgame: StaticGame, only_traj: bool = False) -> StaticSolvingContext:
    """
    Preprocess the game for each player -> Compute all possible actions and outcomes
    """
    available_traj = compute_actions(sgame=sgame)

    if not only_traj:
        # Compute the outcomes for each player action
        tic = perf_counter()
        for player, actions in available_traj.items():
            for traj in actions:
                sgame.get_outcomes(frozendict({player: traj}))
        toc = perf_counter() - tic
        print(f"Preprocess_player: outcomes evaluation time = {toc:.2f} s")

    return get_context(sgame=sgame, actions=available_traj)


def preprocess_full_game(sgame: StaticGame, only_traj: bool = False) -> StaticSolvingContext:
    """
    Preprocess the game -> Compute all possible actions and outcomes for each combination
    """

    available_traj = compute_actions(sgame=sgame)

    if not only_traj:
        # Compute the outcomes for each joint action combination
        tic = perf_counter()
        for joint_traj in set(iterate_dict_combinations(available_traj)):
            sgame.get_outcomes(joint_traj)  # Outcomes are cached inside get_outcomes
        toc = perf_counter() - tic
        print(f"Preprocess_full: outcomes evaluation time = {toc:.2f} s")

    # from multiprocessing import Pool
    # joint_traj_dict = {k: v for k, v in enumerate(set(iterate_dict_combinations(available_traj)))}
    # tic = perf_counter()
    # pool = Pool()
    # # TODO[SIR]: Metric cache is not shared between threads now
    # pool_res = pool.map_async(func=sgame.get_outcomes, iterable=joint_traj_dict.values())
    # pool_res.get()
    # print(f"Outcomes Eval Time = {perf_counter() - tic}s")
    # pool.close()
    # pool.join()

    return get_context(sgame=sgame, actions=available_traj)


def get_context(sgame: StaticGame, actions: Mapping[PlayerName, FrozenSet[Trajectory]]) \
        -> StaticSolvingContext:

    # Similar to get_outcome_preferences_for_players, use SetPreference1 for Poss
    pref: Mapping[PlayerName, Preference[PlayerOutcome]] = {
        name: player.preference for name, player in sgame.game_players.items()
    }

    context = StaticSolvingContext(
        player_actions=actions,
        game_outcomes=sgame.get_outcomes,
        outcome_pref=pref,  # todo I fear here it's missing the monadic preferences but it is fine for now
        solver_params=StaticSolverParams(
            admissible_strategies=PURE_STRATEGIES, strategy_multiple_nash=BAIL_MNE  # this is not used for now
        ),
    )
    return context
