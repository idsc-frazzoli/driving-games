import math
from itertools import product
from typing import FrozenSet, Set, Mapping
from decimal import Decimal as D

from .structures import VehicleState, VehicleActions, VehicleGeometry, TrajectoryParams

__all__ = ["BicycleDynamics"]


class BicycleDynamics:
    v_max: D
    """ Maximum speed [m/s] """

    v_min: D
    """ Minimum speed [m/s] """

    st_max: D
    """ Maximum steering angle [rad] """

    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""

    u_acc: FrozenSet[D]
    """ Possible values of acceleration [m/s2] """

    u_dst: FrozenSet[D]
    """ Possible values of steering rate [rad/s] """

    def __init__(self, params: TrajectoryParams):
        self.v_max = params.v_max
        self.v_min = params.v_min
        self.st_max = params.st_max
        self.vg = params.vg
        self.u_acc = params.u_acc
        self.u_dst = params.u_dst
        if not self.u_acc:
            raise ValueError("No feasible acceleration")
        if not self.u_dst:
            raise ValueError("No feasible steering rate")

    def all_actions(self) -> Set[VehicleActions]:
        res = set()
        for acc, dst in product(self.u_acc, self.u_dst):
            res.add(VehicleActions(acc=acc, dst=dst))
        return res

    def successors(self, x: VehicleState, dt: D, u0: VehicleActions = None) \
            -> Mapping[VehicleActions, VehicleState]:
        """ For each state, returns a dictionary U -> Possible Xs """

        def get_clip(val, lo, hi):
            if lo <= val <= hi:
                return val
            if val < lo:
                return lo
            return hi

        if u0 is None:
            u0 = VehicleActions(acc=D("0"), dst=D("0"))
        u_acc = set([get_clip(val=_+u0.acc, lo=(self.v_min-x.v)/dt, hi=(self.v_max-x.v)/dt) for _ in self.u_acc])
        u_dst = set([get_clip(val=_+u0.dst, lo=(-self.st_max-x.st)/dt, hi=(self.st_max-x.st)/dt) for _ in self.u_dst])

        res = {}
        for acc, dst in product(u_acc, u_dst):
            u = VehicleActions(acc=acc, dst=dst)
            res[u] = self.successor(x, u, dt)
        return res

    def successor(self, x0: VehicleState, u: VehicleActions, dt: D):
        def clip(value, low, high):
            return max(low, min(high, value))

        vf = clip(x0.v + u.acc * dt, low=self.v_min, high=self.v_max)
        stf = clip(x0.st + u.dst * dt, low=-self.st_max, high=self.st_max)
        u_clip = VehicleActions(acc=(vf - x0.v) / dt, dst=(stf - x0.st) / dt)

        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * dt, u_clip)
        ret = x0 + (k1 + k2) * (dt / D("2"))
        return ret

    def dynamics(self, x0: VehicleState, u: VehicleActions) -> VehicleState:

        dx = x0.v
        dr = dx * D(math.tan(x0.st)) / (2 * self.vg.l)
        dy = dr * self.vg.l
        costh = D(math.cos(x0.th + dr / D("2")))
        sinth = D(math.sin(x0.th + dr / D("2")))

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        ret = VehicleState(x=xdot, y=ydot, th=dr, v=u.acc, st=u.dst, t=D("1"))
        return ret
