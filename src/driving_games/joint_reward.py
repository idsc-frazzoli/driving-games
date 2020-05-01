import itertools
from decimal import Decimal as D
from typing import FrozenSet, Mapping, Tuple

import numpy as np

from games import JointRewardStructure, PlayerName
from geometry import SE2, SE2_from_xytheta, xytheta_from_SE2
from .structures import CollisionCost, VehicleActions, VehicleState

__all__ = ["VehicleJointReward"]


class VehicleJointReward(JointRewardStructure[VehicleState, VehicleActions, CollisionCost]):
    def __init__(self, collision_threshold: float):
        self.collision_threshold = collision_threshold

    # @lru_cache(None)
    def is_joint_final_state(self, xs: Mapping[PlayerName, VehicleState]) -> FrozenSet[PlayerName]:
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

    def joint_reward(self, xs: Mapping[PlayerName, VehicleState]) -> Mapping[PlayerName, CollisionCost]:
        players = self.is_joint_final_state(xs)
        if not players:  # pragma: no cover
            raise Exception()
        res = {}
        for p in players:
            res[p] = CollisionCost(xs[p].v)
        return res


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
