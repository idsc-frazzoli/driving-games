from typing import FrozenSet, Mapping

from frozendict import frozendict

from functools import lru_cache
from duckietown_world.utils import memoized_reset

from games import JointRewardStructure, PlayerName
from driving_games.collisions import Collision

from duckie_games.collisions_check import (
    spatial_collision_check_binary_resources_no_rectangles,
    spatial_collision_check_binary_resources_no_rectangles_players_only,
    spatial_collision_check_resources_no_energy_players_only,
    spatial_collision_check_resources_no_energy,
    collision_check_resources_no_energy,
    collision_check_resources_no_energy_players_only,
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

    # todo find suitable collision function
    # @memoized_reset
    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, DuckieState]) -> FrozenSet[PlayerName]:
        res = spatial_collision_check_resources_no_energy_players_only(xs, self.geometries, self.dynamics)

        # only for debugging
        # res = spatial_collision_check_binary_resources_no_rectangles_players_only(xs, self.geometries, self.dynamics)
        return frozenset(res)

    # @memoized_reset
    # @lru_cache(None)
    def joint_reward(self, xs: Mapping[PlayerName, DuckieState]) -> Mapping[PlayerName, Collision]:
        res = spatial_collision_check_resources_no_energy(xs, self.geometries, self.dynamics)

        # only for debugging
        # res = spatial_collision_check_binary_resources_no_rectangles(xs, self.geometries, self.dynamics)
        return res

    def __hash__(self):  # to be able to use lru cache
        return hash(frozendict(self.__dict__))
