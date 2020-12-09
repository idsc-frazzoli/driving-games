from itertools import product
from typing import Dict, Set, FrozenSet, Mapping
from frozendict import frozendict
from time import perf_counter

from games import PlayerName
from possibilities import Poss
from trajectory_game import Trajectory, Metric, MetricEvaluationResult, VehicleState, World, PlayerOutcome, \
    TrajectoryGameOutcome
from trajectory_game.game_def import StaticGame, StaticGamePlayer

__all__ = [
    "JointTrajProfile",
    "TrajectoryGamePlayer",
    "TrajectoryGame",
    "compute_game_outcomes"
]

JointTrajProfile = Mapping[PlayerName, Trajectory]


class TrajectoryGamePlayer(StaticGamePlayer[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


class TrajectoryGame(StaticGame[VehicleState, Trajectory, World, PlayerOutcome]):
    pass


def compute_game_outcomes(traj_game: StaticGame) -> \
        Poss[Mapping[JointTrajProfile, TrajectoryGameOutcome]]:
    """
    Preprocess the game -> Compute all possible actions and outcomes for each combination
    """

    # Generate the trajectories for each player
    all_traj: Dict[PlayerName, Poss[FrozenSet[Trajectory]]] = {}
    for player_name, game_player in traj_game.game_players.items():
        def get_traj_set(state: VehicleState) -> FrozenSet[Trajectory]:
            return game_player.action_set_generator.get_action_set(state=state,
                                                                   world=traj_game.world,
                                                                   player=player_name)

        all_traj[player_name] = traj_game.ps.build(a=game_player.state, f=get_traj_set)

    def build_sets(data_in: Mapping[PlayerName, FrozenSet[Trajectory]]) -> \
            Mapping[JointTrajProfile, TrajectoryGameOutcome]:
        ret: Dict[JointTrajProfile, TrajectoryGameOutcome] = {}
        for joint_traj in (frozendict(zip(data_in.keys(), x))
                           for x in product(*data_in.values())):
            ret[joint_traj] = traj_game.game_outcomes(joint_traj, traj_game.world)
        return frozendict(ret)

    tic = perf_counter()
    traj_outcomes = traj_game.ps.build_multiple(a=all_traj, f=build_sets)
    toc = perf_counter() - tic
    print('Outcomes evaluation time = {} s'.format(toc))
    return traj_outcomes
