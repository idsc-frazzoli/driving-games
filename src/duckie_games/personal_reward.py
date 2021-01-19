from decimal import Decimal as D, localcontext

from games import PersonalRewardStructure
from zuper_commons.types import check_isinstance
from duckie_games.structures import DuckieActions, DuckieCosts, DuckieState


class DuckiePersonalRewardStructureTime(PersonalRewardStructure[DuckieState, DuckieActions, DuckieCosts]):
    max_path: D

    def __init__(self, max_path: D):
        self.max_path = max_path

    def personal_reward_incremental(self, x: DuckieState, u: DuckieActions, dt: D) -> DuckieCosts:
        check_isinstance(x, DuckieState)
        check_isinstance(u, DuckieActions)
        return DuckieCosts(dt)

    def personal_reward_reduce(self, r1: DuckieCosts, r2: DuckieCosts) -> DuckieCosts:
        return r1 + r2

    def personal_reward_identity(self) -> DuckieCosts:
        return DuckieCosts(D(0))

    def personal_final_reward(self, x: DuckieState) -> DuckieCosts:
        check_isinstance(x, DuckieState)
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.max_path - x.x) / x.v

            return DuckieCosts(remaining)

    def is_personal_final_state(self, x: DuckieState) -> bool:
        check_isinstance(x, DuckieState)
        # return x.x > self.max_path
        # fixme why with velocity? this implies step size of 1?
        # why not return x.x >= self.max_path?
        return x.x + x.v > self.max_path
