from dataclasses import replace
from decimal import Decimal
from typing import FrozenSet, Mapping

from dg_commons import PlayerName, Timestamp, valmap
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.collisions_check import collision_check
from driving_games.structures import VehicleActions, VehicleTrackState
from games import JointRewardStructure

__all__ = ["VehicleJointReward"]


def _find_parent_state(x: VehicleTrackState, dt: Decimal) -> VehicleTrackState:
    return replace(x, x=x.x - x.v * dt)


class VehicleJointReward(JointRewardStructure[VehicleTrackState, VehicleActions, CollisionReportPlayer]):
    def __init__(self, game_dt: Timestamp, geometries: Mapping[PlayerName, VehicleGeometry], col_check_dt: float):
        self.game_dt = Decimal(game_dt)
        self.geometries = geometries
        self.col_check_dt = col_check_dt

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleTrackState]) -> FrozenSet[PlayerName]:
        # az todo here a better interface would consider the transition rather than the current state

        initial = valmap()

        res = collision_check(xs, self.geometries)
        return frozenset(res)

    def joint_reward(self, xs: Mapping[PlayerName, VehicleTrackState]) -> Mapping[PlayerName, CollisionReportPlayer]:
        res = collision_check(xs, self.geometries)
        return res


# az this is temporary not used
# class IndividualJointReward(JointRewardStructure[VehicleState, VehicleActions, CollisionReportPlayer]):
#     def __init__(
#             self,
#             collision_threshold: float,
#             geometries: Mapping[PlayerName, VehicleGeometry],
#             caring_players: List[PlayerName],
#     ):
#         self.collision_threshold = collision_threshold
#         self.geometries = geometries
#         self.caring_players = caring_players  # az not sure what is this for?!
#
#     # @lru_cache(None)
#     def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
#         res = collision_check(xs, self.geometries)
#         # filtered_res = set()
#         # for player in res:
#         #     if player in self.caring_players:
#         #         filtered_res.add(player)
#         # del res[player]
#         # col = res[player]
#         # res[player] = Collision(col.location, False, D(0), D(0))
#         return frozenset(res)
#
#     def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, Collision]:
#         res = collision_check(xs, self.geometries)
#         for player, cost in res.items():
#             if player in self.caring_players:
#                 res[player] = Collision(IMPACT_FRONT, False, D(0), D(0))
#         return res
