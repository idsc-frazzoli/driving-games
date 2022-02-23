from decimal import Decimal as D, localcontext
from math import inf

from zuper_commons.types import check_isinstance

from dg_commons import Timestamp
from games import PersonalRewardStructure, StageIdx
from .structures import VehicleActions, VehicleTimeCost, VehicleTrackState

__all__ = ["VehiclePersonalRewardStructureTime"]


class VehiclePersonalRewardStructureTime(PersonalRewardStructure[VehicleTrackState, VehicleActions, VehicleTimeCost]):
    goal_progress: D

    def __init__(self, goal_progress: D, maximum_depth: StageIdx = +inf):
        self.goal_progress = goal_progress
        self.maximum_depth = maximum_depth

    def personal_reward_incremental(self, x: VehicleTrackState, u: VehicleActions, dt: Timestamp) -> VehicleTimeCost:
        check_isinstance(x, VehicleTrackState)
        check_isinstance(u, VehicleActions)
        return VehicleTimeCost(dt)

    def personal_reward_reduce(self, r1: VehicleTimeCost, r2: VehicleTimeCost) -> VehicleTimeCost:
        return r1 + r2

    def personal_reward_identity(self) -> VehicleTimeCost:
        return VehicleTimeCost(D(0))

    def personal_final_reward(self, x: VehicleTrackState) -> VehicleTimeCost:
        check_isinstance(x, VehicleTrackState)

        with localcontext() as ctx:
            ctx.prec = 3
            try:
                remaining = (self.goal_progress - x.x) / x.v
            except:
                remaining = D(0)
            return VehicleTimeCost(remaining)

    def is_personal_final_state(self, x: VehicleTrackState, depth: StageIdx = 0) -> bool:
        check_isinstance(x, VehicleTrackState)
        return x.x >= self.goal_progress or depth >= self.maximum_depth
