from functools import partial
from typing import Dict, Set, FrozenSet, Mapping, Tuple, List
from time import perf_counter

from dataclasses import dataclass
from frozendict import frozendict

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE
from games.utils import iterate_dict_combinations
from possibilities import Poss
from preferences import Preference

from .structures import VehicleState, VehicleGeometry
from .paths import Trajectory
from .trajectory_world import TrajectoryWorld
from .metrics_def import PlayerOutcome
from .game_def import Game, GamePlayer, SolvingContext, SolvedGameNode, StaticSolverParams, EXP_ACCOMP, \
    AntichainComparison

__all__ = [
    "JointPureTraj",
    "TrajectoryGamePlayer",
    "TrajectoryGame",
    "LeaderFollowerPrefs",
    "LeaderFollowerGameSolvingContext",
    "LeaderFollowerGame",
    "SolvedTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "LeaderFollowerGameNode",
    "SolvedLeaderFollowerGame",
    "preprocess_full_game",
    "preprocess_player",
]

JointPureTraj = Mapping[PlayerName, Trajectory]


@dataclass
class TrajectoryGamePlayer(GamePlayer[VehicleState, Trajectory,
                                      TrajectoryWorld, PlayerOutcome,
                                      VehicleGeometry]):
    pass


@dataclass
class TrajectoryGame(Game[VehicleState, Trajectory,
                          TrajectoryWorld, PlayerOutcome,
                          VehicleGeometry]):
    pass


@dataclass
class LeaderFollowerPrefs:
    leader: PlayerName
    follower: PlayerName
    pref_leader: Preference
    prefs_follower: Poss[Preference]
    antichain_comparison: AntichainComparison


@dataclass
class LeaderFollowerGameSolvingContext(SolvingContext):
    lf: LeaderFollowerPrefs


@dataclass
class LeaderFollowerGame(TrajectoryGame):
    lf: LeaderFollowerPrefs


@dataclass(frozen=True, unsafe_hash=True)
class SolvedTrajectoryGameNode(SolvedGameNode[Trajectory, PlayerOutcome]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]


@dataclass(unsafe_hash=True)
class LeaderFollowerGameNode:
    nodes: SolvedTrajectoryGame
    agg_lead_outcome: PlayerOutcome


@dataclass(unsafe_hash=True)
class SolvedLeaderFollowerGame:
    lf: LeaderFollowerPrefs
    games: Mapping[Trajectory, Mapping[Preference, LeaderFollowerGameNode]]
    """ All possible game results for both players """
    best_leader_actions: Mapping[Preference, Set[Trajectory]]


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
    if isinstance(sgame, LeaderFollowerGame):
        ac_comp = sgame.lf.antichain_comparison
    else:
        ac_comp = EXP_ACCOMP

    solver_params = StaticSolverParams(admissible_strategies=PURE_STRATEGIES,
                                       strategy_multiple_nash=BAIL_MNE,
                                       antichain_comparison=ac_comp, use_best_response=True)
    kwargs = {
        "player_actions": actions, "game_outcomes": sgame.get_outcomes,
        "outcome_pref": pref, "solver_params": solver_params
    }
    if isinstance(sgame, LeaderFollowerGame):
        context = LeaderFollowerGameSolvingContext(**kwargs, lf=sgame.lf)
    else:
        context = SolvingContext(**kwargs)
    return context
