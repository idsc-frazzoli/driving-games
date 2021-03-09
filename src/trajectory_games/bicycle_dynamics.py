import math
from itertools import product
from typing import FrozenSet, Set, Mapping
from decimal import Decimal as D

from .structures import VehicleState, VehicleActions, VehicleGeometry, TrajectoryParams

__all__ = ["BicycleDynamics"]


class BicycleDynamics:
    v_max: float
    """ Maximum speed [m/s] """

    v_min: float
    """ Minimum speed [m/s] """

    st_max: float
    """ Maximum steering angle [rad] """

    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""

    u_acc: FrozenSet[float]
    """ Possible values of acceleration [m/s2] """

    u_dst: FrozenSet[float]
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

    @staticmethod
    def get_clipped(val: float, lo: float, hi: float) -> float:
        """ Get value clipped between limits """
        if lo <= val <= hi:
            return val
        if val < lo:
            return lo
        return hi

    def get_feasible_acc(self, x: VehicleState, dt: D, u0: VehicleActions) -> Set[float]:
        """ Get feasible accelerations for current state with mean u0 """
        dt_f = float(dt)
        u_acc = set([self.get_clipped(val=_+u0.acc,
                                      lo=(self.v_min-x.v)/dt_f,
                                      hi=(self.v_max-x.v)/dt_f)
                     for _ in self.u_acc])
        return u_acc

    def get_feasible_dst(self, x: VehicleState, dt: D, u0: VehicleActions) -> Set[float]:
        """ Get feasible steering rates for current state with mean u0 """
        dt_f = float(dt)
        u_dst = set([self.get_clipped(val=_+u0.dst,
                                      lo=(-self.st_max-x.st)/dt_f,
                                      hi=(self.st_max-x.st)/dt_f)
                     for _ in self.u_dst])
        return u_dst

    def successors(self, x: VehicleState, dt: D, u0: VehicleActions = None) \
            -> Mapping[VehicleActions, VehicleState]:
        """ For each state, returns a dictionary U -> Possible Xs """

        if u0 is None:
            u0 = VehicleActions(acc=0.0, dst=0.0)
        u_acc = self.get_feasible_acc(x=x, dt=dt, u0=u0)
        u_dst = self.get_feasible_dst(x=x, dt=dt, u0=u0)

        res = {}
        for acc, dst in product(u_acc, u_dst):
            u = VehicleActions(acc=acc, dst=dst)
            res[u] = self.successor_forward(x, u, dt)
        return res

    def successor(self, x0: VehicleState, u: VehicleActions, dt: D):
        """ Perform RK2 integration to propagate state using actions for time dt """
        def clip(value, low, high):
            return max(low, min(high, value))
        dt_f = float(dt)
        vf = clip(x0.v + u.acc * dt_f, low=self.v_min, high=self.v_max)
        stf = clip(x0.st + u.dst * dt_f, low=-self.st_max, high=self.st_max)
        u_clip = VehicleActions(acc=(vf - x0.v) / dt_f, dst=(stf - x0.st) / dt_f)

        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * dt_f, u_clip)
        ret = x0 + (k1 + k2) * (dt_f / 2.0)
        return ret

    def successor_forward(self, x0: VehicleState, u: VehicleActions, dt: D):
        """ Perform Euler forward integration to propagate state using actions for time dt """
        dt_f = float(dt)
        v0, st0 = x0.v, x0.st
        x0.v += u.acc * dt_f
        x0.st += u.dst * dt_f
        tol = 1e-3
        if not self.v_min - tol <= x0.v <= self.v_max + tol:
            print("Velocity outside limits")
        if not -self.st_max - tol <= x0.st <= self.st_max + tol:
            print(f"Steering = {x0.st}, outside limits")
        u = VehicleActions(acc=0.0, dst=0.0)

        k1 = self.dynamics(x0, u)
        ret = x0 + k1 * dt_f
        x0.v, x0.st = v0, st0
        return ret

    def dynamics(self, x0: VehicleState, u: VehicleActions) -> VehicleState:
        """ Get rate of change of states for given control inputs """
        dx = x0.v
        dr = dx * math.tan(x0.st) / (2.0 * self.vg.l)
        dy = dr * self.vg.l
        costh = math.cos(x0.th + dr / 2.0)
        sinth = math.sin(x0.th + dr / 2.0)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        ret = VehicleState(x=xdot, y=ydot, th=dr, v=u.acc, st=u.dst, t=D("1"))
        return ret
