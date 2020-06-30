from typing import FrozenSet, Mapping, List

from games import JointRewardStructure, PlayerName
from .collisions import Collision
from .collisions_check import collision_check
from .structures import VehicleActions, VehicleGeometry, VehicleState

__all__ = ["VehicleJointReward"]


class VehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, Collision]):
    def __init__(
        self, collision_threshold: float, geometries: Mapping[PlayerName, VehicleGeometry],
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
        res = collision_check(xs, self.geometries)
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, Collision]:
        res = collision_check(xs, self.geometries)
        return res

class IndividualJointReward(JointRewardStructure[VehicleState, VehicleActions, Collision]):
    def __init__(
        self, collision_threshold: float, geometries: Mapping[PlayerName, VehicleGeometry],alone_players: List[PlayerName]
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries
        self.alone_players = alone_players

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
        res = collision_check(xs, self.geometries)
#        for player in self.alone_players:
#            res[player].active = False
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, Collision]:
        res = collision_check(xs, self.geometries)
#        for player in self.alone_players:
#            res[player].active = False
        return res