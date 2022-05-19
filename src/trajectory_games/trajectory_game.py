from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import random
from time import perf_counter
from typing import Dict, FrozenSet, Mapping, Optional, Set, Tuple, Union

import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.scenario.trajectory import State
from commonroad_dc.feasibility import feasibility_checker
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from frozendict import frozendict

# from commonroad_challenge.situational_traj_generator import feasibility_check
from dg_commons import iterate_dict_combinations, PlayerName, logger
from dg_commons.seq.sequence import DgSampledSequence, Timestamp
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.metrics_structures import PlayerEvaluatedMetrics
from games import BAIL_MNE, PURE_STRATEGIES
from possibilities import Poss
from preferences import Preference
from . import TrajectoryGenerator
from .game_def import (
    AntichainComparison,
    EXP_ACCOMP,
    Game,
    GamePlayer,
    SolvedGameNode,
    SolvingContext,
    StaticSolverParams,
)
# from .paths import Trajectory, TrajectoryGraph
from dg_commons.planning import Trajectory, TrajectoryGraph
from dg_commons.sim.models.vehicle import VehicleState
from .trajectory_world import TrajectoryWorld

__all__ = [
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
    "get_only_context",
    "get_context_and_graphs"
]

"""@dataclass
class SingleActionPlayer(
    GamePlayer[VehicleState, Trajectory, TrajectoryWorld, PlayerEvaluatedMetrics, VehicleGeometry]
):
    pass"""


@dataclass
class TrajectoryGamePlayer(
    GamePlayer[VehicleState, Trajectory, TrajectoryWorld, PlayerEvaluatedMetrics, VehicleGeometry]
):
    pass


@dataclass
class TrajectoryGame(Game[VehicleState, Trajectory, TrajectoryWorld, PlayerEvaluatedMetrics, VehicleGeometry]):
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
class SolvedTrajectoryGameNode(SolvedGameNode[Trajectory, PlayerEvaluatedMetrics]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]


@dataclass(unsafe_hash=True)
class LeaderFollowerGameNode:
    nodes: SolvedTrajectoryGame
    agg_lead_outcome: PlayerEvaluatedMetrics


@dataclass(unsafe_hash=True)
class SolvedLeaderFollowerGame:
    """Single stage solution of the game"""

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
    """Single stage of the game"""

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
    """Entire recursive multistage game as a sequence of stages"""

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
    """Generate the trajectories for each player (i.e. get the available actions)"""
    print("\nGenerating Trajectories:")
    available_traj: Dict[PlayerName, FrozenSet[Trajectory]] = {}
    for player_name, game_player in sgame.game_players.items():
        # In the future can be extended to uncertain initial state
        states = game_player.state.support()
        assert len(states) == 1, states
        available_traj[player_name] = game_player.actions_generator.get_actions(
            state=next(iter(states)),
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


# todo: TEST this
def sample_trajectories(
        all_trajs: Mapping[PlayerName, FrozenSet[Trajectory]],
        n_trajs_max: Optional[Union[int, Mapping[PlayerName, int]]] = None,
        method: str = "unif"
):
    """
    Subsample trajectories uniformly at random for each player.
    :param all_trajs:   Set of all trajectories for all players.
    :param n_trajs_max: Maximum number of trajectories to sample for each player.
    :param method:      how to sample. It can be uniform ("unif" or "uniform") or try to maximize the variability of the
                        generated trajectories ("variance" or "var).
    :return:            Mapping from players to sampled subset of trajectories.
    """
    if n_trajs_max is None:
        return all_trajs

    subset_trajs: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
    random.seed(0)  # todo: fix this and take seed from SimContext
    if method.lower() == "unif" or method.lower() == "uniform":
        for pname, player_trajs in all_trajs.items():
            if type(n_trajs_max) == int:
                n = n_trajs_max
            else:
                n = n_trajs_max[pname]

            if len(list(player_trajs)) <= n:
                p_trajs = player_trajs
            else:
                p_trajs = random.sample(list(player_trajs), k=n)

            subset_trajs[pname] = frozenset(p_trajs)
    elif method.lower() == "variance" or method.lower() == "var":
        for pname, player_trajs in all_trajs.items():
            if type(n_trajs_max) == int:
                n = n_trajs_max
            else:
                n = n_trajs_max[pname]

            if len(list(player_trajs)) <= n:
                p_trajs = player_trajs
            else:
                # add half the number of required subsampled trajectories
                p_trajs = set(random.sample(list(player_trajs), k=int(n / 2)))
                new_candidate_trajs = set(list(player_trajs)) - p_trajs
                # select one trajectory at random from subsampled trajectories up to now
                while len(p_trajs) < n:
                    p_compare_new = random.sample(list(p_trajs), k=1)
                    candidate_trajs = list(new_candidate_trajs)
                    mse = [p_compare_new.squared_error(new_traj) for new_traj in candidate_trajs]
                    index = mse.index((max(mse)))
                    p_trajs.add(candidate_trajs[index])
                    new_candidate_trajs.remove(candidate_trajs[index])

                    # for not yet added trajectories, compute MSE with p_compare_new

                    # add the trajectory with the highest MSE -> favour diversity

                    # repeat until we have the required number of elements

            subset_trajs[pname] = frozenset(p_trajs)

    else:
        raise NotImplementedError

    return subset_trajs


# todo: next three function to clean up when integrating Situational Trajectory Generator
def convert_to_cr_state(vehicle_state: VehicleState, time_step: int = 0) -> State:
    return State(
        position=np.array([vehicle_state.x, vehicle_state.y]),
        orientation=vehicle_state.theta,
        velocity=vehicle_state.vx,
        steering_angle=vehicle_state.delta,
        time_step=time_step,
    )


from commonroad.scenario.trajectory import Trajectory as CR_Trajectory


def feasibility_check(traj: Trajectory, vehicle_dynamics: VehicleDynamics, dt: Timestamp) -> bool:
    cr_traj_states = []
    values = traj.values
    for i, state in enumerate(values):
        current_state = convert_to_cr_state(state)
        current_state.time_step = i
        cr_traj_states.append(current_state)
    cr_traj = CR_Trajectory(initial_time_step=0, state_list=cr_traj_states)
    # check feasibility of planned trajectory for the given vehicle model and dynamics
    feasible, _ = feasibility_checker.trajectory_feasibility(cr_traj, vehicle_dynamics, dt)
    return feasible


# todo: integrate Situational Trajectory Generator better
def filter_actions(trajectories: FrozenSet[Trajectory], n_actions: int = 10) -> FrozenSet[Trajectory]:
    """
    Filter actions through a set of criteria, e.g. feasibility
    :return:
    """
    vehicle_dynamics = VehicleDynamics.KS(VehicleType.FORD_ESCORT)

    subset_trajs = set()
    remaining_trajs = set(trajectories)

    while len(subset_trajs) < n_actions and len(remaining_trajs) != 0:
        cand_traj = random.sample(remaining_trajs, 1)[0]
        dt = cand_traj.timestamps[1] - cand_traj.timestamps[0]
        remaining_trajs.remove(cand_traj)
        # todo: account for feasibility
        # feasible = feasibility_check(cand_traj, vehicle_dynamics, dt)
        # if feasible:
        # print("found one feasible trajectory")
        subset_trajs.add(cand_traj)

    # print("Total number of trajectories: " + str(len(subset_trajs)))
    return frozenset(subset_trajs)


def get_context_and_graphs(
        game: TrajectoryGame,
        sampling_method: str,
        pad_trajectories: float = False,
        max_n_traj: Optional[Union[int, Mapping[PlayerName, int]]] = None
) -> Tuple[SolvingContext, Mapping[PlayerName, FrozenSet[TrajectoryGraph]]]:
    """
    Construct solving context and return trajectory graphs for all players.
    :param game:                Trajectory Game
    :param sampling_method:     Which method to use to subsample trajectories from all those available.
    :param pad_trajectories:    Extend all trajectories that are shorter than the longest one, keeping the
                                vehicle state constant.
    :param max_n_traj:          Maximum number of trajectories to return
    :return:                    Solving Context and Trajectory graph
    """

    def generate_trajectory_graphs(game: TrajectoryGame) -> Mapping[PlayerName, FrozenSet[TrajectoryGraph]]:

        """Generate graph of trajectories and commands for each player (i.e. get the available actions)"""
        logger.info(f"Generating Trajectories")
        traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]] = {}
        for player_name, game_player in game.game_players.items():
            if isinstance(game_player.actions_generator, TrajectoryGenerator):
                states = game_player.state.support()
                assert len(states) == 1, states
                traj_graphs[player_name] \
                    = game_player.actions_generator.get_actions(state=list(states)[0], return_graphs=True)
            else:
                raise RuntimeError("No trajectory generator found for " + str(player_name))

        logger.info(f"Trajectory generation finished.")
        return traj_graphs

    def get_context(sgame: TrajectoryGame,
                    actions: Mapping[PlayerName, FrozenSet[Trajectory]]) -> SolvingContext:

        pref: Mapping[PlayerName, Preference[PlayerEvaluatedMetrics]] = {
            name: player.preference for name, player in sgame.game_players.items()
        }

        kwargs = {
            "player_actions": actions,
            "game_outcomes": sgame.get_outcomes,
            "outcome_pref": pref,
            "solver_params": None,
        }

        return SolvingContext(**kwargs)

    traj_graphs: Mapping[PlayerName, FrozenSet[TrajectoryGraph]] = generate_trajectory_graphs(game=game)
    all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}

    # retrieve all possible trajectories stored in trajectory graphs for each player
    for player_name, game_player in game.game_players.items():
        all_trajectories_p: Set[Trajectory] = set()
        for graph in traj_graphs[player_name]:
            # all_trajectories_p |= graph.get_all_trajectories()
            all_trajectories_p |= graph.get_all_transitions()
            if player_name == PlayerName("Ego"):
                accept_only_feasible = False
                # todo: account for feasibility!
            else:
                accept_only_feasible = False  # todo: workaround for DEU_Cologne-40_6

            subset_trajs_p = filter_actions(trajectories=frozenset(all_trajectories_p),
                                            n_actions=max_n_traj[player_name])
            all_trajectories[player_name] = frozenset(subset_trajs_p)

    # # subsample trajectories at random to limit action number
    # subset_trajs = sample_trajectories(all_trajs=all_trajectories, n_trajs_max=max_n_traj, method=sampling_method)
    # max_time: Timestamp = -999.0
    # if pad_trajectories:
    #     for pname, trajectories_set in subset_trajs.items():
    #         for traj in trajectories_set:
    #             max_time = max(traj.timestamps[-1], max_time)
    #
    #     if max_time > 0.0:
    #         for pname, trajectories_set in subset_trajs.items():
    #             new_set = set()
    #             for traj in trajectories_set:
    #                 if max_time > traj.timestamps[-1]:
    #                     new_set.add(traj.pad_to_time(t_final=max_time, dt=1))
    #                 else:
    #                     new_set.add(traj)
    #             subset_trajs[pname] = frozenset(new_set)

    for joint_traj in set(iterate_dict_combinations(all_trajectories)):
        game.get_outcomes(joint_traj)

    return get_context(sgame=game, actions=all_trajectories), traj_graphs


def get_only_context(
        sgame: TrajectoryGame,
        actions: Mapping[PlayerName, FrozenSet[Trajectory]]
) -> SolvingContext:
    """
    Construct solving context and return trajectory graphs for all players.
    :param game:        Trajectory Game
    :param actions:     Trajectories for each player
    :return:            Solving Context
    """

    def get_context(sgame: TrajectoryGame,
                    actions: Mapping[PlayerName, FrozenSet[Trajectory]]) -> SolvingContext:
        pref: Mapping[PlayerName, Preference[PlayerEvaluatedMetrics]] = {
            name: player.preference for name, player in sgame.game_players.items()
        }

        kwargs = {
            "player_actions": actions,
            "game_outcomes": sgame.get_outcomes,
            "outcome_pref": pref,
            "solver_params": None,
        }

        return SolvingContext(**kwargs)

    for joint_traj in set(iterate_dict_combinations(actions)):
        sgame.get_outcomes(joint_traj)

    return get_context(sgame=sgame, actions=actions)


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
    pref: Mapping[PlayerName, Preference[PlayerEvaluatedMetrics]] = {
        name: player.preference for name, player in sgame.game_players.items()
    }
    if isinstance(sgame, LeaderFollowerGame):
        ac_comp = sgame.lf.antichain_comparison
    else:
        ac_comp = EXP_ACCOMP
    # todo [LEON] are these correct? Not used right now
    solver_params = StaticSolverParams(
        admissible_strategies=PURE_STRATEGIES,
        strategy_multiple_nash=BAIL_MNE,
        dt=1.,
        factorization_algorithm="TEST",  # todo: remove
        use_factorization=False,
        n_simulations=5,
        extra=False,
        max_depth=3

        # antichain_comparison=ac_comp,
        # use_best_response=True,
    )

    kwargs = {
        "player_actions": actions,
        "game_outcomes": sgame.get_outcomes,
        "outcome_pref": pref,
        "solver_params": solver_params,
    }
    if isinstance(sgame, LeaderFollowerGame):
        context = LeaderFollowerGameSolvingContext(**kwargs, lf=deepcopy(sgame.lf))
    else:
        context = SolvingContext(**kwargs)
    return context
