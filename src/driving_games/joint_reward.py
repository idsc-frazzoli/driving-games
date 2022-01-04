from dataclasses import replace
from decimal import Decimal
from typing import FrozenSet, Mapping, Dict

from commonroad.scenario.lanelet import LaneletNetwork

from dg_commons import PlayerName, Timestamp, DgSampledSequence, RJ
from dg_commons.maps import DgLanelet
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games.collisions_check import collision_check
from driving_games.structures import VehicleActions, VehicleTrackState
from games import JointRewardStructure

__all__ = ["VehicleJointReward"]

from games.game_def import JointTransition


def _find_parent_state(x: VehicleTrackState, dt: Decimal) -> VehicleTrackState:
    return replace(x, x=x.x - x.v * dt)


class VehicleJointReward(JointRewardStructure[VehicleTrackState, VehicleActions, CollisionReportPlayer]):
    def __init__(
        self,
        geometries: Mapping[PlayerName, VehicleGeometry],
        ref_lanes: Mapping[PlayerName, DgLanelet],
        col_check_dt: Timestamp,
        lanelet_network: LaneletNetwork,
    ):
        assert geometries.keys() == ref_lanes.keys()
        self.geometries = geometries
        self.ref_lane = ref_lanes
        self.col_check_dt = col_check_dt
        self.lanelet_network = lanelet_network

    def joint_reward_incremental(self, txs: JointTransition) -> Mapping[PlayerName, RJ]:
        pass  # todo

    def joint_reward_reduce(self, r1: RJ, r2: RJ) -> RJ:
        pass  # todo

    def joint_reward_identity(self) -> RJ:
        pass  # todo

    def is_joint_final_transition(
        self, txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
    ) -> FrozenSet[PlayerName]:
        res = self.joint_final_reward(txs)
        return frozenset(res)

    def joint_final_reward(
        self, txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
    ) -> Mapping[PlayerName, CollisionReportPlayer]:
        global_xs: Dict[PlayerName:VehicleState] = {}
        for p in txs:

            def to_vehicle_state(tx: VehicleTrackState):
                t = tx.to_global_pose(self.ref_lane[p])
                return VehicleState(x=t.p[0], y=t.p[1], theta=t.theta, vx=float(tx.v), delta=0)

            global_xs[p] = txs[p].transform_values(to_vehicle_state, VehicleState)
        res = collision_check(global_xs, self.geometries, self.col_check_dt, self.lanelet_network)
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
