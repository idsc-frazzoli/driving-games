from dataclasses import replace
from decimal import Decimal
from typing import Dict, FrozenSet, Mapping

from dg_commons import DgSampledSequence, fd, PlayerName, Timestamp
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import JointRewardStructure
from games.game_def import JointState, JointTransition
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

    def joint_reward_incremental(self, txs: JointTransition) -> Mapping[PlayerName, VehicleJointCost]:
        res: Dict[PlayerName, VehicleJointCost] = {}
        if len(txs) > 1:
            # ####### temp
            # k0 = list(txs.keys())[0]
            # dt = txs[k0].get_end()
            # interaction_graph = Graph()
            # interaction_graph.clear()
            # resources_used = {p: self.players_dynamics[p].get_shared_resources(txs[p].values[0], dt=dt) for p in txs}
            # interaction_graph.add_nodes_from(resources_used)
            # for p1, p2 in combinations(resources_used, 2):
            #     intersects = bool(resources_used[p1] & resources_used[p2])
            #     if intersects:
            #         interaction_graph.add_edge(p1, p2)
            # interacting_subsets = frozenset(map(frozenset, connected_components(interaction_graph)))
            # for p_subset in interacting_subsets:
            #    for p in p_subset:if len(p_subset) > 1:
            # #######
            global_xs: Dict[PlayerName:VehicleState] = {}
            for p in txs:  # p_subset:

                def to_vehicle_state(tx: VehicleTrackState):
                    t = tx.to_global_pose(self.ref_lane[p])
                    return VehicleState(x=t.p[0], y=t.p[1], theta=t.theta, vx=float(tx.v), delta=0)

                global_xs[p] = txs[p].transform_values(to_vehicle_state, VehicleState)
            res.update(
                joint_simple_collision_cost(fd(global_xs), self.geometries, self.col_check_dt, self.min_safety_distance)
            )
            # else:
            #     res.update({p: self.joint_reward_identity() for p in p_subset})
        else:
            res.update({p: self.joint_reward_identity() for p in txs})
        return res

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
