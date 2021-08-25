from copy import deepcopy
from functools import partial
from typing import Dict, Set, FrozenSet, Mapping, Optional
from time import perf_counter

from dataclasses import dataclass
from frozendict import frozendict

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE
from games.utils import iterate_dict_combinations
from possibilities import Poss
from preferences import Preference

from dg_commons.sequence import Timestamp, DgSampledSequence
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
    "LeaderFollowerParams",
    "LeaderFollowerGameSolvingContext",
    "LeaderFollowerGame",
    "SolvedTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "LeaderFollowerGameNode",
    "SolvedLeaderFollowerGame",
    "LeaderFollowerGameStage",
    "SolvedRecursiveLeaderFollowerGame",
    "preprocess_full_game",
    "preprocess_player",
]

JointPureTraj = Mapping[PlayerName, Trajectory]
""" Joint action of all players in the game """


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
class LeaderFollowerParams:
    leader: PlayerName
    follower: PlayerName
    """ Names of the player """

    pref_leader: Preference
    prefs_follower_est: Poss[Preference]
    """ Pref of leader and all possible prefs of follower to solve the game """

    antichain_comparison: AntichainComparison
    """ Antichain comparison method """

    solve_time: Timestamp
    simulation_step: Timestamp
    """ Timestep for each solution and simulation [s]"""

    terminal_progress: float
    """ Fraction of leader lane to finish game """

    update_prefs: bool
    """ Update estimated preferences of follower online or not """

    pref_follower_real: Optional[Preference] = None
    """ Real preference of follower used to simulate the game """


@dataclass
class LeaderFollowerGameSolvingContext(SolvingContext):
    lf: LeaderFollowerParams


@dataclass
class LeaderFollowerGame(TrajectoryGame):
    lf: LeaderFollowerParams


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
    """ Single stage solution of the game """

    lf: LeaderFollowerParams
    """ Game params"""
    games: Mapping[Trajectory, Mapping[Preference, LeaderFollowerGameNode]]
    """ All possible game results for both players """
    best_leader_actions: Mapping[Preference, Set[Trajectory]]
    """ Best actions of leader for each estimated preference of follower """
    meet_leader_actions: Set[Trajectory]
    """ Best actions of leader for meet of estimated prefs of follower """


@dataclass
class LeaderFollowerGameStage:
    """ Single stage of the game """

    lf: LeaderFollowerParams
    """ Game params """
    context: LeaderFollowerGameSolvingContext
    """ Current context to solve the game """

    lf_game: SolvedLeaderFollowerGame
    game_node: SolvedTrajectoryGameNode
    """ Game solution """

    best_responses_pred: Set[Trajectory]
    """ Predicted best responses of follower """
    states: Mapping[PlayerName, Poss[VehicleState]]
    """ Initial states of both players """
    time: Timestamp
    """ Time of the stage solved [s] """


@dataclass
class SolvedRecursiveLeaderFollowerGame:
    """ Entire recursive multistage game as a sequence of stages """

    lf: LeaderFollowerParams
    stages: DgSampledSequence[LeaderFollowerGameStage]
    aggregated_node: SolvedTrajectoryGameNode
    """ Final aggregated actions and outcomes of players """


def compute_outcomes(iterable, sgame: Game):
    key, joint_traj_in = iterable
    get_outcomes = partial(sgame.get_outcomes, world=sgame.world)
    ps = sgame.ps
    return key, ps.build(ps.unit(joint_traj_in), f=get_outcomes)


def compute_actions(sgame: Game) -> Mapping[PlayerName, FrozenSet[Trajectory]]:
    """ Generate the trajectories for each player (i.e. get the available actions) """
    print("\nGenerating Trajectories:")
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
        context = LeaderFollowerGameSolvingContext(**kwargs, lf=deepcopy(sgame.lf))
    else:
        context = SolvingContext(**kwargs)
    return context
