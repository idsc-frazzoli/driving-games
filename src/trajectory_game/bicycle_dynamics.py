import itertools
import math
from typing import FrozenSet, Set, Mapping

from .structures import VehicleState, VehicleActions, VehicleGeometry


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

    def __init__(self, v_max: float, v_min: float, st_max: float,
                 vg: VehicleGeometry, u_acc: FrozenSet[float],
                 u_dst: FrozenSet[float]):
        self.v_max = v_max
        self.v_min = v_min
        self.st_max = st_max
        self.vg = vg
        self.u_acc = u_acc
        self.u_dst = u_dst

    def all_actions(self) -> Set[VehicleActions]:
        res = set()
        for acc, dst in itertools.product(self.u_acc, self.u_dst):
            res.add(VehicleActions(acc=acc, dst=dst))
        return res

    def successors(self, x: VehicleState, dt: float) -> Mapping[VehicleActions, VehicleState]:
        """ For each state, returns a dictionary U -> Possible Xs """

        # only allow inputs with feasible final vel, st
        u_acc = [_ for _ in self.u_acc if self.v_min <= x.v + _ * dt <= self.v_max]
        u_dst = [_ for _ in self.u_dst if -self.st_max <= x.st + _ * dt <= self.st_max]

        res = {}
        for acc, dst in itertools.product(u_acc, u_dst):
            u = VehicleActions(acc=acc, dst=dst)
            res[u] = self.successor(x, u, dt)
        return res

    def successor(self, x0: VehicleState, u: VehicleActions, dt: float):
        def clip(value, low, high):
            return max(low, min(high, value))

        vf = clip(x0.v + u.acc * dt, low=self.v_min, high=self.v_max)
        stf = clip(x0.st + u.dst * dt, low=-self.st_max, high=self.st_max)
        u_clip = VehicleActions(acc=(vf-x0.v)/dt, dst=(stf-x0.st)/dt)

        alpha = 1.
        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * (dt * alpha), u_clip)
        ret = x0 + k1 * (dt * (1 - 0.5 / alpha)) + k2 * (dt * (0.5 / alpha))
        return ret

    def dynamics(self, x0: VehicleState, u: VehicleActions) -> VehicleState:

        dx = x0.v
        dr = dx * math.tan(x0.st) / (self.vg.lf + self.vg.lr)
        dy = dr * self.vg.lr
        costh = math.cos(x0.th + dr/2.)
        sinth = math.sin(x0.th + dr/2.)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        ret = VehicleState(x=xdot, y=ydot, th=dr, v=u.acc, st=u.dst, t=1.)
        return ret
