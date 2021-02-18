from typing import FrozenSet, Mapping

from duckietown_world.utils import memoized_reset

from games import JointRewardStructure, PlayerName
from driving_games.collisions import Collision

from duckie_games.collisions_check import (
    collision_check_resources_no_energy,
    collision_check_players_only_resources,
    collision_check_rectangle_energy
)
from duckie_games.structures import DuckieActions, DuckieGeometry, DuckieState
from duckie_games.duckie_dynamics import DuckieDynamics


class DuckieJointReward(JointRewardStructure[DuckieState, DuckieActions, Collision]):
    def __init__(
        self,
        collision_threshold: float,
        geometries: Mapping[PlayerName, DuckieGeometry],
        dynamics: Mapping[PlayerName, DuckieDynamics],
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries
        self.dynamics = dynamics

    #@lru_cache(None)
    @memoized_reset
    def is_joint_final_state(self, xs: Mapping[PlayerName, DuckieState]) -> FrozenSet[PlayerName]:
        res = collision_check_players_only_resources(xs, self.geometries, self.dynamics)
        return frozenset(res)

    #@lru_cache(None)
    @memoized_reset
    def joint_reward(self, xs: Mapping[PlayerName, DuckieState]) -> Mapping[PlayerName, Collision]:
        res = collision_check_resources_no_energy(xs, self.geometries, self.dynamics)
        return res
