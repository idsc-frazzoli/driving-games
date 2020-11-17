import itertools
from typing import Mapping, FrozenSet, Tuple, List, Set
from decimal import Decimal as D, localcontext

from zuper_commons.types import ZNotImplementedError, check_isinstance

from bayesian_driving_games.structures import PlayerType, AGGRESSIVE, CAUTIOUS, NEUTRAL
from driving_games import (
    VehicleState,
    VehicleActions,
    Collision,
    VehicleGeometry,
    frozendict,
    IMPACT_FRONT,
    IMPACT_SIDES,
    VehicleCosts,
)
from driving_games.collisions_check import (
    sample_x,
    rectangle_from_pose,
    a_caused_collision_with_b,
    collision_check,
)
from games import JointRewardStructure, PlayerName, PersonalRewardStructure


def bayesian_collision_check(
    poses: Mapping[PlayerName, VehicleState],
    geometries: Mapping[PlayerName, VehicleGeometry],
    p1_types,
    p2_types,
) -> Mapping[Tuple[PlayerType], Mapping[PlayerName, Collision]]:
    """

    :param poses: The state of each player
    :param geometries: Geometries
    :param p1_types: The types of player 1
    :param p2_types: The types of player 2
    :return: For each type combination, for each player a collision object
    """

    type_combinations = list(itertools.product(p1_types, p2_types))
    dt = D(0.5)
    n = 2
    if len(poses) == 1:
        return frozendict({})
    if len(poses) > 2:
        raise ZNotImplementedError(players=set(poses))
    res1 = {}
    res2 = {}
    res = {}
    p1, p2 = list(poses)
    s1 = poses[p1]
    s2 = poses[p2]
    g1 = geometries[p1]
    g2 = geometries[p2]

    x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
    x2s = sample_x(s2.x, s2.v, dt=dt, n=n)

    for x1, x2 in zip(x1s, x2s):
        pc1 = rectangle_from_pose(s1.ref, x1, g1)
        pc2 = rectangle_from_pose(s2.ref, x2, g2)

        # did p1 collide with p2?
        p1_caused = a_caused_collision_with_b(pc1, pc2)
        p2_caused = a_caused_collision_with_b(pc2, pc1)

        p1_active = p1_caused
        p2_active = p2_caused
        if p1_caused and p2_caused:
            # head-on collision
            i1 = i2 = IMPACT_FRONT
            vs = s1.v * g1.mass + s2.v * g2.mass
            energy_received_1 = vs
            energy_received_2 = vs
            energy_given_1 = vs
            energy_given_2 = vs
            pass
        elif p1_caused:
            i1 = IMPACT_FRONT
            i2 = IMPACT_SIDES
            energy_received_1 = D(0)
            energy_received_2 = s1.v * g1.mass
            energy_given_1 = s1.v * g1.mass
            energy_given_2 = D(0)
        elif p2_caused:
            i1 = IMPACT_SIDES
            i2 = IMPACT_FRONT
            energy_received_2 = D(0)
            energy_received_1 = s1.v * g1.mass
            energy_given_2 = s1.v * g1.mass
            energy_given_1 = D(0)
        else:
            continue

        c1 = Collision(i1, p1_active, energy_received_1, energy_given_1)
        c2 = Collision(i2, p2_active, energy_received_2, energy_given_2)
        # c5 = Collision(i2, True, energy_received_2, energy_given_2)
        # c3 = Collision(i2, p2_active, D(0), D(0))
        # c4 = Collision(i1, False, D(0), D(0))
        res1 = {p1: c1, p2: c2}  # cautious
        res2 = {p1: c1, p2: c2}  # aggressive
        res[type_combinations[0]] = res1
        res[type_combinations[1]] = res2
        return res

    return {}


class BayesianVehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, Collision]):
    """
    Standard joint reward, but with a collision cost for each player in each type combination.
    """

    def __init__(
        self, collision_threshold: float, geometries: Mapping[PlayerName, VehicleGeometry], p1_types, p2_types
    ):
        self.collision_threshold = collision_threshold
        self.geometries = geometries
        self.p1_types = p1_types
        self.p2_types = p2_types

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
        res = collision_check(xs, self.geometries)
        return frozenset(res)

    def joint_reward(
        self, xs: Mapping[PlayerName, VehicleState]
    ) -> Mapping[Tuple[PlayerType], Mapping[PlayerName, Collision]]:
        res = bayesian_collision_check(xs, self.geometries, self.p1_types, self.p2_types)
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

    def __init__(self, max_path: D, p1_types: Set[PlayerType]):
        self.max_path = max_path
        # todo only my (of the player) possible types are needed?!
        self.p_types = p1_types

    def personal_reward_incremental(
        self, x: VehicleState, u: VehicleActions, dt: D
    ) -> Mapping[Tuple[PlayerType, PlayerType], VehicleCosts]:
        """
        #fixme check return argument... shouldn't it be Mapping[PlayerType, VehicleCosts]?
        :param x: The state of the player
        :param u: The action of the player
        :param dt: Timestep
        :return: For each type combination a Cost (VehicleCosts is defined as "duration", but can be anything really.
        """
        check_isinstance(x, VehicleState)
        check_isinstance(u, VehicleActions)
        res = dict.fromkeys([AGGRESSIVE, CAUTIOUS, NEUTRAL])
        res[AGGRESSIVE] = VehicleCosts((dt + D(0.1)) * (dt + D(0.1)))
        res[CAUTIOUS] = VehicleCosts(abs(u.accel))
        res[NEUTRAL] = VehicleCosts((dt + D(0.1)) * (dt + D(0.1)))  # fixme is this a proper cost?
        # fixme RETURN res[BayesianVehicleState].my_type?
        return res

    def personal_reward_reduce(self, r1: VehicleCosts, r2: VehicleCosts) -> VehicleCosts:
        return r1 + r2

    def personal_reward_identity(self) -> VehicleCosts:
        return VehicleCosts(D(0))

    def personal_final_reward(self, x: VehicleState) -> Mapping[Tuple[PlayerType, PlayerType], VehicleCosts]:
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
            tc = list(itertools.product(self.p1_types, self.p2_types))
            res[tc[0]] = VehicleCosts(remaining)
            res[tc[1]] = VehicleCosts(remaining)
            return res

    def is_personal_final_state(self, x: VehicleState) -> bool:
        check_isinstance(x, VehicleState)
        # return x.x > self.max_path

        return x.x + x.v > self.max_path


class BayesianVehiclePersonalRewardStructureSimple(
    PersonalRewardStructure[VehicleState, VehicleActions, VehicleCosts]
):

    """
    This is a second reward structure very similar to the one above, so that different reward structures for different
    players can be chosen.
    """

    max_path: D
    # todo "oh my..." this shall go once the above is adjusted

    def __init__(self, max_path: D, p1_types: List[PlayerType], p2_types: List[PlayerType]):
        self.max_path = max_path
        self.p1_types = p1_types
        self.p2_types = p2_types

    def personal_reward_incremental(
        self, x: VehicleState, u: VehicleActions, dt: D
    ) -> Mapping[Tuple[PlayerType, PlayerType], VehicleCosts]:
        tc = list(itertools.product(self.p1_types, self.p2_types))
        check_isinstance(x, VehicleState)
        check_isinstance(u, VehicleActions)
        res = {}
        res[tc[0]] = VehicleCosts((dt + D(0.1)) * (dt + D(0.1)))  # cautious
        res[tc[1]] = VehicleCosts(dt)  # aggressive
        return res

    def personal_reward_reduce(self, r1: VehicleCosts, r2: VehicleCosts) -> VehicleCosts:
        return r1 + r2

    def personal_reward_identity(self) -> VehicleCosts:
        return VehicleCosts(D(0))

    def personal_final_reward(self, x: VehicleState) -> Mapping[Tuple[PlayerType, PlayerType], VehicleCosts]:
        check_isinstance(x, VehicleState)
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.max_path - x.x) / x.v
            res = {}
            tc = list(itertools.product(self.p1_types, self.p2_types))
            res[tc[0]] = VehicleCosts(remaining)
            res[tc[1]] = VehicleCosts(remaining)
            return res

    def is_personal_final_state(self, x: VehicleState) -> bool:
        check_isinstance(x, VehicleState)
        # return x.x > self.max_path

        return x.x + x.v > self.max_path
