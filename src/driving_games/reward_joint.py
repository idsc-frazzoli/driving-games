from dataclasses import replace
from decimal import Decimal
from typing import FrozenSet, Mapping, Dict, Optional

from commonroad.scenario.lanelet import LaneletNetwork

from dg_commons import PlayerName, Timestamp, DgSampledSequence, fd
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.time import time_function
from driving_games import VehicleJointCost, VehicleSafetyDistCost
from driving_games.collisions_check import joint_collision_cost_simple
from driving_games.structures import VehicleActions, VehicleTrackState
from games import JointRewardStructure
from games.game_def import JointTransition

__all__ = ["VehicleJointReward"]


def _find_parent_state(x: VehicleTrackState, dt: Decimal) -> VehicleTrackState:
    return replace(x, x=x.x - x.v * dt)


class VehicleJointReward(JointRewardStructure[VehicleTrackState, VehicleActions, VehicleJointCost]):
    def __init__(
        self,
        geometries: Mapping[PlayerName, VehicleGeometry],
        ref_lanes: Mapping[PlayerName, DgLanelet],
        col_check_dt: Timestamp,
        min_safety_distance: float,
        lanelet_network: Optional[LaneletNetwork] = None,
    ):
        assert geometries.keys() == ref_lanes.keys()
        self.geometries = fd(geometries)
        self.ref_lane = fd(ref_lanes)
        self.col_check_dt = col_check_dt
        self.min_safety_distance = min_safety_distance
        self.lanelet_network = lanelet_network

    def joint_reward_incremental(self, txs: JointTransition) -> Mapping[PlayerName, VehicleJointCost]:
        return self.joint_final_reward(txs)

    def joint_reward_reduce(self, r1: VehicleJointCost, r2: VehicleJointCost) -> VehicleJointCost:
        return r1 + r2

    def joint_reward_identity(self) -> VehicleJointCost:
        return VehicleJointCost(VehicleSafetyDistCost(0))

    def is_joint_final_transition(
        self, txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
    ) -> FrozenSet[PlayerName]:
        res = self.joint_final_reward(txs)
        return frozenset({p for p in res if res[p].collision is not None})

    @time_function
    def joint_final_reward(
        self, txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
    ) -> Mapping[PlayerName, VehicleJointCost]:
        global_xs: Dict[PlayerName:VehicleState] = {}
        # todo need to test if this transform works as expected
        for p in txs:

            def to_vehicle_state(tx: VehicleTrackState):
                t = tx.to_global_pose(self.ref_lane[p])
                return VehicleState(x=t.p[0], y=t.p[1], theta=t.theta, vx=float(tx.v), delta=0)

            global_xs[p] = txs[p].transform_values(to_vehicle_state, VehicleState)
        res = joint_collision_cost_simple(fd(global_xs), self.geometries, self.col_check_dt, self.min_safety_distance)
        return res
