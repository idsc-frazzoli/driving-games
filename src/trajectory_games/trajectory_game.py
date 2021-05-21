from functools import partial
from typing import Dict, Set, FrozenSet, Mapping, Generic, Tuple, TypeVar
from time import perf_counter

from dataclasses import dataclass
from frozendict import frozendict

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE, P
from games.utils import iterate_dict_combinations
from preferences import Preference

from .structures import VehicleState, VehicleGeometry
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld
from .metrics_def import PlayerOutcome
from .game_def import Game, GamePlayer, SolvingContext, SolvedGameNode, StaticSolverParams, EXP_ACCOMP

__all__ = [
    "JointPureTraj",
    "TrajectoryGamePlayer",
    "TrajectoryGame",
    "SolvedTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "LeaderFollowerNode",
    "SolvedLeaderFollowerGameNode",
    "SolvedLeaderFollowerGame",
    "preprocess_full_game",
    "preprocess_player",
]

JointPureTraj = Mapping[PlayerName, Trajectory]


class TrajectoryGamePlayer(GamePlayer[VehicleState, Trajectory,
                                      TrajectoryWorld, PlayerOutcome,
                                      VehicleGeometry]):
    pass


class TrajectoryGame(Game[VehicleState, Trajectory,
                          TrajectoryWorld, PlayerOutcome,
                          VehicleGeometry]):
    pass


class SolvedTrajectoryGameNode(SolvedGameNode[Trajectory, PlayerOutcome]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]


X = TypeVar("X")


@dataclass(unsafe_hash=True)
class LeaderFollowerNode(Generic[X]):
    predicted: X
    """ Predicted by leader """
    simulated: X
    """ Calculated by follower """


@dataclass(unsafe_hash=True)
class SolvedLeaderFollowerGameNode(Generic[P]):
    players: Tuple[PlayerName, PlayerName]
    """ Leader and Follower """
    games: LeaderFollowerNode[SolvedTrajectoryGame]
    """ All possible game results for both players - Predicted, Simulated (P,S) """
    leader_game: LeaderFollowerNode[SolvedTrajectoryGameNode]
    """ Aggregated game solution for leader (P,S) """
    player_pref: LeaderFollowerNode[Mapping[PlayerName, Preference[P]]]
    """ Player preferences (P,S) """


SolvedLeaderFollowerGame = Set[SolvedLeaderFollowerGameNode]


def compute_outcomes(iterable, sgame: Game):
    key, joint_traj_in = iterable
    get_outcomes = partial(sgame.get_outcomes, world=sgame.world)
    ps = sgame.ps
    return key, ps.build(ps.unit(joint_traj_in), f=get_outcomes)


def compute_actions(sgame: Game) -> Mapping[PlayerName, FrozenSet[Trajectory]]:
    """ Generate the trajectories for each player (i.e. get the available actions) """
    available_traj: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, game_player in sgame.game_players.items():
        # In the future can be extended to uncertain initial state
        states = game_player.state.support()
        assert len(states) == 1, states
        available_traj[player_name] = game_player.actions_generator.get_actions_static(
            state=next(iter(states)), world=sgame.world, player=player_name
        )
    return available_traj


def preprocess_player(sgame: Game, only_traj: bool = False) -> SolvingContext:
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


def preprocess_full_game(sgame: Game, only_traj: bool = False) -> SolvingContext:
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


def get_context(sgame: Game, actions: Mapping[PlayerName, FrozenSet[Trajectory]]) -> SolvingContext:
    # Similar to get_outcome_preferences_for_players, use SetPreference1 for Poss
    pref: Mapping[PlayerName, Preference[PlayerOutcome]] = {
        name: player.preference for name, player in sgame.game_players.items()
    }

    context = SolvingContext(
        player_actions=actions,
        game_outcomes=sgame.get_outcomes,
        outcome_pref=pref,  # todo I fear here it's missing the monadic preferences but it is fine for now
        solver_params=StaticSolverParams(
            admissible_strategies=PURE_STRATEGIES, strategy_multiple_nash=BAIL_MNE,
            antichain_comparison=EXP_ACCOMP, use_best_response=True
        ),
    )
    return context
