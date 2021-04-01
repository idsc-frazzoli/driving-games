from functools import partial
from typing import Dict, Set, FrozenSet, Mapping, Tuple, Optional, Hashable
from time import perf_counter

from frozendict import frozendict

from games import PlayerName, PURE_STRATEGIES, BAIL_MNE, JointState
from games.utils import iterate_dict_combinations
from preferences import Preference

from .structures import VehicleState, VehicleGeometry
from .paths import Transition, Trajectory, Action
from .trajectory_world import TrajectoryWorld
from .metrics_def import PlayerOutcome, TrajGameOutcome
from .game_def import Game, StaticGamePlayer, StaticSolvingContext, \
    SolvedGameNode, StaticSolverParams, DynamicGamePlayer

__all__ = [
    "JointAction",
    "JointPureTraj",
    "JointTrans",
    "StaticTrajectoryGamePlayer",
    "DynamicTrajectoryGamePlayer",
    "StaticTrajectoryGame",
    "DynamicTrajectoryGame",
    "SolvedTrajectoryGameNode",
    "SolvedStaticTrajectoryGameNode",
    "SolvedDynamicTrajectoryGameNode",
    "SolvedTrajectoryGame",
    "SolvedStaticTrajectoryGame",
    "SubgameSolutions",
    "preprocess_full_game",
    "preprocess_player",
]

JointAction = Mapping[PlayerName, Action]
JointPureTraj = Mapping[PlayerName, Trajectory]
JointTrans = Mapping[PlayerName, Transition]


class StaticTrajectoryGamePlayer(StaticGamePlayer[VehicleState, Trajectory,
                                                  TrajectoryWorld, PlayerOutcome,
                                                  VehicleGeometry]):
    pass


class DynamicTrajectoryGamePlayer(DynamicGamePlayer[VehicleState, Transition,
                                                    TrajectoryWorld, PlayerOutcome,
                                                    VehicleGeometry]):
    pass


class StaticTrajectoryGame(Game[VehicleState, Trajectory,
                                TrajectoryWorld, PlayerOutcome,
                                VehicleGeometry]):
    pass


class DynamicTrajectoryGame(Game[VehicleState, Transition,
                                 TrajectoryWorld, PlayerOutcome,
                                 VehicleGeometry]):
    pass


class SolvedTrajectoryGameNode(SolvedGameNode[Action, PlayerOutcome]):
    pass


class SolvedStaticTrajectoryGameNode(SolvedGameNode[Trajectory, PlayerOutcome]):
    pass


class SolvedDynamicTrajectoryGameNode(SolvedGameNode[Transition, PlayerOutcome]):
    pass


SolvedTrajectoryGame = Set[SolvedTrajectoryGameNode]
SolvedStaticTrajectoryGame = Set[SolvedStaticTrajectoryGameNode]


class SubgameSolutions:
    Action_outcome = Tuple[JointAction, TrajGameOutcome]
    Anti_chain = FrozenSet[JointPureTraj]
    Traj_dict = Dict[JointState, Anti_chain]
    best_traj: Traj_dict

    def __init__(self, traj_dict: Traj_dict = None):
        self.best_traj = traj_dict if traj_dict is not None else {}

    def __getitem__(self, item: JointState) -> Optional[Anti_chain]:
        if not isinstance(item, Hashable):
            item = frozendict(item)
        if item in self.best_traj:
            return self.best_traj[item]
        return None

    def get_trajectories(self, joint_act: JointAction) -> Anti_chain:
        joint_state: Mapping[PlayerName, VehicleState] = \
            {p: trans.at(trans.get_end()) for p, trans in joint_act.items()}
        return self.append(joint_act=joint_act, best=self[joint_state])

    @staticmethod
    def append(joint_act: JointAction, best: Anti_chain) -> Anti_chain:
        ret: Set[JointPureTraj] = set()
        if best is None:
            joint_traj: JointPureTraj = frozendict({p: trans + None for p, trans in joint_act.items()})
            ret.add(joint_traj)
            return frozenset(ret)

        for joint_best in best:
            joint_traj: JointPureTraj = \
                frozendict({player: joint_act[player] + joint_best[player] for player in joint_best.keys()})
            ret.add(joint_traj)
        return frozenset(ret)

    @staticmethod
    def accumulate_indiv(m1: PlayerOutcome, m2: PlayerOutcome) -> PlayerOutcome:
        if m2 is None:
            return m1
        if m1.keys() != m2.keys():
            raise ValueError(f"Keys don't match - {m1.keys(), m2.keys()}")
        outcome: PlayerOutcome = {k: m1[k] + m2[k] for k in m1.keys()}
        return outcome


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


def preprocess_player(sgame: Game, only_traj: bool = False) -> StaticSolvingContext:
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


def preprocess_full_game(sgame: Game, only_traj: bool = False) -> StaticSolvingContext:
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


def get_context(sgame: Game, actions: Mapping[PlayerName, FrozenSet[Trajectory]]) \
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
