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
