import itertools
from decimal import Decimal as D, localcontext
from typing import FrozenSet as ASet, Mapping, Optional, Tuple

import numpy as np
from geometry import SE2, SE2_from_xytheta, xytheta_from_SE2
from zuper_commons.types import check_isinstance
from zuper_typing import debug_print

from driving_games.structures import CollisionCost, VehicleActions, VehicleState
from games import (
    Combined,
    JointRewardStructure,
    PersonalRewardStructure,
    PlayerName,
)
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    LexicographicPreference,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferredTol,
)

# noinspection PyTypeChecker

#
# X_ = VehicleState
# U_ = VehicleActions
# Y_ = VehicleObservation
# RP_ = D

# CollisionCost = Coll

# RJ_ = CollisionCost


class VehiclePersonalRewardStructureTime(PersonalRewardStructure):
    max_path: D

    def __init__(self, max_path: D):
        self.max_path = max_path

    def personal_reward_incremental(self, x: VehicleState, u: VehicleActions, dt: D) -> D:
        return dt

    def personal_reward_reduce(self, r1: D, r2: D) -> D:
        return r1 + r2

    def personal_final_reward(self, x: VehicleState) -> D:
        # assert self.is_personal_final_state(x)

        with localcontext() as ctx:
            ctx.prec = 2
            remaining = (self.max_path - x.x) / x.v

            return remaining

    def is_personal_final_state(self, x: VehicleState) -> bool:
        # return x.x > self.max_path

        return x.x + x.v > self.max_path


class CollisionPreference(Preference[Optional[CollisionCost]]):
    def __init__(self):
        self.p = SmallerPreferredTol(D(0))

    def get_type(self):
        return Optional[CollisionCost]

    def compare(self, a: Optional[CollisionCost], b: Optional[CollisionCost]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None:
            return FIRST_PREFERRED
        if b is None:
            return SECOND_PREFERRED
        res = self.p.compare(a.v, b.v)
        assert res in COMP_OUTCOMES, (res, self.p)
        return res

    def __repr__(self):
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "CollisionPreference:\n " + debug_print(d)


class VehiclePreferencesCollTime(Preference[Combined[CollisionCost, D]]):
    def __init__(self):
        self.collision = CollisionPreference()
        self.time = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.time))

    def get_type(self):
        return Combined[CollisionCost, D]

    def __repr__(self):
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[CollisionCost, D], b: Combined[CollisionCost, D]
    ) -> ComparisonOutcome:
        check_isinstance(a, Combined)
        check_isinstance(b, Combined)
        # ct_a = (a.joint, a.personal)
        # ct_b = (b.joint, b.personal)
        if a.joint is None and b.joint is None:
            return self.time.compare(a.personal, b.personal)
        else:
            return self.collision.compare(a.joint, b.joint)
        # # res = self.lexi.compare(ct_a, ct_b)
        # assert res in COMP_OUTCOMES, (res, self.lexi)
        # return res


def SE2_from_VehicleState(s: VehicleState):
    p = SE2_from_xytheta([float(s.x), 0, 0])
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    return SE2.multiply(ref, p)


def pose_diff(a, b):
    S = SE2
    return S.multiply(S.inverse(a), b)


def sample_from_traj(s: VehicleState, dt: D, n: int) -> Tuple[Tuple[float, float], ...]:
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    res = []
    for i in range(-n, +n + 1):
        x2 = s.x + s.v * D(i) * dt
        p = SE2_from_xytheta([float(x2), 0, 0])
        p2 = SE2.multiply(ref, p)
        x1, y1, _ = xytheta_from_SE2(p2)
        res.append((x1, y1))
    return tuple(res)


class VehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, CollisionCost]):
    def __init__(self, collision_threshold: float):
        self.collision_threshold = collision_threshold

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> ASet[PlayerName]:
        if len(xs) == 1:
            return frozenset()
        if len(xs) != 2:
            raise NotImplementedError(len(xs))
        s1, s2 = list(xs.values())
        mind = 1000
        dt = D(0.5)
        n = 2
        samples1 = sample_from_traj(s1, dt=dt, n=n)
        samples2 = sample_from_traj(s2, dt=dt, n=n)
        for (x1, y1), (x2, y2) in itertools.product(samples1, samples2):
            dist = np.hypot(x1 - x2, y1 - y2)
            mind = min(mind, dist)
        # d = pose_diff(c1, c2)
        # x, y, _ = xytheta_from_SE2(d)
        # dist = np.hypot(x, y)
        # logger.info(c1=xytheta_from_SE2(c1), c2=xytheta_from_SE2(c2), dist=dist)
        if mind < self.collision_threshold:
            return frozenset(xs)
        else:
            return frozenset()

    def joint_reward(
        self, xs: Mapping[PlayerName, VehicleState]
    ) -> Mapping[PlayerName, CollisionCost]:
        players = self.is_joint_final_state(xs)
        if not players:
            raise Exception()
        res = {}
        for p in players:
            res[p] = CollisionCost(xs[p].v)
        return res
