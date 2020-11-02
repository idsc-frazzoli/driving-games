from typing import FrozenSet, Mapping, List
from decimal import Decimal as D

from games import JointRewardStructure, PlayerName
from .collisions import Collision, IMPACT_FRONT
from .collisions_check import collision_check
from .structures import VehicleActions, VehicleGeometry, VehicleState

__all__ = ["VehicleJointReward"]


class VehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, Collision]):
    def __init__(
        self,
        collision_threshold: float,
        geometries: Mapping[PlayerName, VehicleGeometry],
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
        self,
        collision_threshold: float,
        geometries: Mapping[PlayerName, VehicleGeometry],
        caring_players: List[PlayerName],
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries
        self.caring_players = caring_players

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
        res = collision_check(xs, self.geometries)
        # filtered_res = set()
        # for player in res:
        #     if player in self.caring_players:
        #         filtered_res.add(player)
        # del res[player]
        # col = res[player]
        # res[player] = Collision(col.location, False, D(0), D(0))
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, Collision]:
        res = collision_check(xs, self.geometries)
        for player, cost in res.items():
            if player in self.caring_players:
                res[player] = Collision(IMPACT_FRONT, False, D(0), D(0))
        return res
