from dataclasses import replace
from decimal import Decimal
from typing import Dict, Mapping, FrozenSet

from dg_commons import DgSampledSequence, fd, PlayerName, Timestamp
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import JointRewardStructure
from games.game_def import JointTransition, JointState
from . import VehicleTrackDynamics
from .collisions import VehicleJointCost, VehicleSafetyDistCost
from .collisions_check import joint_simple_collision_cost
from .structures import VehicleActions, VehicleTrackState

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
        players_dynamics: Mapping[PlayerName, VehicleTrackDynamics],
    ):
        assert geometries.keys() == ref_lanes.keys()
        self.geometries = fd(geometries)
        self.ref_lane = fd(ref_lanes)
        self.col_check_dt = col_check_dt
        self.min_safety_distance = min_safety_distance
        self.players_dynamics = players_dynamics
        self._cached_dts = None
        """ Assumes transition always of the same length"""

    def joint_reward_incremental(self, txs: JointTransition) -> Mapping[PlayerName, VehicleJointCost]:
        res: Dict[PlayerName, VehicleJointCost] = {}
        if len(txs) > 1:
            if self._cached_dts is None:
                k0 = list(txs.keys())[0]
                t1_end, t1_start = txs[k0].get_end(), txs[k0].get_start()
                # we up-sample the transition according to col_dt from the end going backwards
                n1 = int((t1_end - t1_start) / self.col_check_dt)
                ts = [t1_end - i * self.col_check_dt for i in range(n1 + 1)]
                # but we evaluate them forward in time (for early exit)
                ts.reverse()
                self._cached_dts = tuple(ts)

            upsampled_txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
            upsampled_txs = fd(
                {
                    p: DgSampledSequence[VehicleTrackState](
                        timestamps=self._cached_dts, values=(txs[p].at_interp(t) for t in self._cached_dts)
                    )
                    for p in txs
                }
            )
            res_ = joint_simple_collision_cost(
                transitions=upsampled_txs,
                geometries=self.geometries,
                ref_lane=self.ref_lane,
                col_dt=self.col_check_dt,
                min_safety_dist=self.min_safety_distance,
            )

            res.update(res_)
        else:
            res.update({p: self.joint_reward_identity() for p in txs})

        return fd(res)

    def joint_reward_reduce(self, r1: VehicleJointCost, r2: VehicleJointCost) -> VehicleJointCost:
        return r1 + r2

    def joint_reward_identity(self) -> VehicleJointCost:
        return VehicleJointCost(VehicleSafetyDistCost(0))

    def is_joint_final_transition(
        self, txs: Mapping[PlayerName, DgSampledSequence[VehicleTrackState]]
    ) -> FrozenSet[PlayerName]:
        res = self.joint_reward_incremental(txs)
        return frozenset({p for p in res if res[p].collision is not None})

    def joint_final_reward(self, xs: JointState[VehicleTrackState]) -> Mapping[PlayerName, VehicleJointCost]:
        """No explicit joint final cost.
        The ending cost is already added in the incremental cost of the last transition."""
        return fd({p: VehicleJointCost(VehicleSafetyDistCost(0)) for p in xs if xs[p].has_collided})

    def is_joint_final_states(self, xs: JointState[VehicleTrackState]) -> FrozenSet[PlayerName]:
        """No explicit joint final cost. The ending cost is already added in the incremental cost."""
        return frozenset({p for p in xs if xs[p].has_collided})
