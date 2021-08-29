import itertools
from typing import Mapping as M, FrozenSet, Tuple, Set
from decimal import Decimal as D, localcontext

from zuper_commons.types import check_isinstance

from _tmp.bayesian_driving_games.structures import PlayerType, AGGRESSIVE, CAUTIOUS, NEUTRAL, BayesianGamePlayer
from driving_games import (
    VehicleState,
    VehicleActions,
    Collision,
    VehicleGeometry,
    VehicleCosts,
)
from driving_games.collisions_check import (
    collision_check,
)
from games import JointRewardStructure, PlayerName, PersonalRewardStructure


class BayesianVehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, Collision]):
    """
    Standard joint reward, but with a collision cost for each player in each type combination.
    """

    def __init__(
        self,
        collision_threshold: float,
        geometries: M[PlayerName, VehicleGeometry],
        players: M[PlayerName, BayesianGamePlayer],
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries
        self.players = players

    # @lru_cache(None)
    def is_joint_final_state(self, xs: M[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
        res = collision_check(xs, self.geometries)
        return frozenset(res)

    def joint_reward(self, xs: M[PlayerName, VehicleState]) -> M[PlayerName, M[PlayerName, Collision]]:
        # todo this is utterly wrong
        res: M[PlayerType] = {}
        for ptype in self.players:
            res[ptype] = collision_check(xs, self.geometries)
        return res


class BayesianVehiclePersonalRewardStructureScalar(
    PersonalRewardStructure[VehicleState, VehicleActions, VehicleCosts]
    # fixme PersonalRewardStructure[BayesianVehicleState, VehicleActions, VehicleCosts]
):
    """
    Similar to the normal Driving Games personal reward, but each type combination gives a different payoff.
    These are coded in the functions `personal_reward_incremental` and `personal_final_reward`.
    """

    max_path: D

    def __init__(self, max_path: D, p_types: Set[PlayerType]):
        self.max_path = max_path
        # todo only my (of the player) possible types are needed?!
        self.p_types = p_types

    def personal_reward_incremental(
        self, x: VehicleState, u: VehicleActions, dt: D
    ) -> M[Tuple[PlayerType, PlayerType], VehicleCosts]:
        """
        #fixme check return argument... shouldn't it be M[PlayerType, VehicleCosts]?
        :param x: The state of the player
        :param u: The action of the player
        :param dt: Timestep
        :return: For each type combination a Cost (VehicleCosts is defined as "duration", but can be anything really.
        """
        check_isinstance(x, VehicleState)
        check_isinstance(u, VehicleActions)
        possible_types = [t for t in self.p_types if t in [AGGRESSIVE, CAUTIOUS, NEUTRAL]]
        res = dict.fromkeys(possible_types)
        for t in possible_types:
            if t == AGGRESSIVE:
                res[t] = VehicleCosts((dt + D(0.1)) * (dt + D(0.1)))
            elif t == CAUTIOUS:
                res[t] = VehicleCosts(abs(u.accel))
            elif t == NEUTRAL:
                res[t] = VehicleCosts((dt + D(0.1)) * (dt + D(0.1)))
            else:
                msg = f'Type of player "{t}" is unrecognized'
                raise NotImplementedError(msg)
        return res

    def personal_reward_reduce(self, r1: VehicleCosts, r2: VehicleCosts) -> VehicleCosts:
        return r1 + r2

    def personal_reward_identity(self) -> VehicleCosts:
        return VehicleCosts(D(0))

    def personal_final_reward(self, x: VehicleState) -> M[Tuple[PlayerType, PlayerType], VehicleCosts]:
        """

        :param x: The state of the agent
        :return: For each type combination a final reward in state x.
        """
        check_isinstance(x, VehicleState)
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.max_path - x.x) / x.v
            res = {}
            # todo fixme
            tc = list(itertools.product(self.p1_types, self.p2_types))
            res[tc[0]] = VehicleCosts(remaining)
            res[tc[1]] = VehicleCosts(remaining)
            return res

    def is_personal_final_state(self, x: VehicleState) -> bool:
        check_isinstance(x, VehicleState)
        # return x.x > self.max_path

        return x.x + x.v > self.max_path
