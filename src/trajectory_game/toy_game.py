from itertools import product
from typing import Dict, Set

from games import PlayerName
from trajectory_game import Trajectory, Rule, RuleEvaluationResult, VehicleState, World
from trajectory_game.game_def import OpenLoopGame, OpenLoopGamePlayer
from typing import Mapping as M


class JointTrajProfile(object):
    pass


def trajectory_set_generator(VehicleState, World) -> Set[Trajectory]:
    # todo
    pass


class TrajGamePlayer(OpenLoopGamePlayer[VehicleState, Set[Trajectory], World]):

    # todo
    pass


def compute_outcomes(traj_game: OpenLoopGame) -> "Outcome":
    """
    # Preprocess trajectory game, for each trajectory combination we compute all the outcomes
    :param traj_game:
    :return:
    """
    # this part generates teh trajectories for each player
    available_traj: Dict[PlayerName, Set[Trajectory]] = {}
    for player_name, game_player in traj_game.game_players.items():
        available_traj[player_name] = game_player.action_set_generator(traj_game.world, game_player.state)

    traj_outcomes: Dict[JointTrajProfile, M[Rule, M[PlayerName, RuleEvaluationResult]]] = {}
    # product(*available_traj)
    # [[a,b],[a,b]] -> [(a,a,),(a,b),(b,a),(b,b)]
    for joint_traj in product(*available_traj):
        traj_outcomes[joint_traj] = traj_game.game_outcomes(joint_traj, traj_game.world)
