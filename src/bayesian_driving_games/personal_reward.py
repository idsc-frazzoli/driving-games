from decimal import Decimal as D, localcontext

import numpy as np

from games import PersonalRewardStructure
from geometry import SE2, SE2_from_xytheta
from zuper_commons.types import check_isinstance
from .structures import VehicleActions, VehicleCosts, BayesianVehicleState, PlayerType

__all__ = ["VehiclePersonalRewardStructureTime"]


class VehiclePersonalRewardStructureTime(PersonalRewardStructure[BayesianVehicleState, VehicleActions, VehicleCosts]):
    max_path: D

    def __init__(self, max_path: D):
        self.max_path = max_path

    def personal_reward_incremental(self, x: BayesianVehicleState, u: VehicleActions, dt: D) -> VehicleCosts:
        check_isinstance(x, BayesianVehicleState)
        # check_isinstance(u, VehicleActions)
        return VehicleCosts(dt)

    def personal_reward_reduce(self, r1: VehicleCosts, r2: VehicleCosts) -> VehicleCosts:
        return r1+r2

    def personal_reward_identity(self) -> VehicleCosts:
        return VehicleCosts(D(0))

    def personal_final_reward(self, x: BayesianVehicleState) -> VehicleCosts:
        check_isinstance(x, BayesianVehicleState)
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.max_path-x.x)/x.v

            return VehicleCosts(remaining)

    def is_personal_final_state(self, x: BayesianVehicleState) -> bool:
        check_isinstance(x, BayesianVehicleState)
        # return x.x > self.max_path

        return x.x+x.v > self.max_path


def SE2_from_VehicleState(s: BayesianVehicleState):
    p = SE2_from_xytheta([float(s.x), 0, 0])
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    return SE2.multiply(ref, p)
