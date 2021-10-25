from decimal import Decimal as D, localcontext

from zuper_commons.types import check_isinstance

from games import PersonalRewardStructure
from .structures import VehicleActions, VehicleCosts, VehicleTrackState

__all__ = ["VehiclePersonalRewardStructureTime"]


class VehiclePersonalRewardStructureTime(PersonalRewardStructure[VehicleTrackState, VehicleActions, VehicleCosts]):
    goal_progress: D

    def __init__(self, goal_progress: D):
        self.goal_progress = goal_progress

    def personal_reward_incremental(self, x: VehicleTrackState, u: VehicleActions, dt: D) -> VehicleCosts:
        check_isinstance(x, VehicleTrackState)
        check_isinstance(u, VehicleActions)
        return VehicleCosts(dt)

    def personal_reward_reduce(self, r1: VehicleCosts, r2: VehicleCosts) -> VehicleCosts:
        return r1 + r2

    def personal_reward_identity(self) -> VehicleCosts:
        return VehicleCosts(D(0))

    def personal_final_reward(self, x: VehicleTrackState) -> VehicleCosts:
        check_isinstance(x, VehicleTrackState)
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.goal_progress - x.x) / x.v
            return VehicleCosts(remaining)

    def is_personal_final_state(self, x: VehicleTrackState) -> bool:
        check_isinstance(x, VehicleTrackState)
        return x.x >= self.goal_progress
