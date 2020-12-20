import math
from itertools import product
from typing import FrozenSet, Set, Mapping
from decimal import Decimal as D

from games import Dynamics
from driving_games import Rectangle, make_rectangle
from .structures import VehicleState, VehicleActions, VehicleGeometry, TrajectoryParams

__all__ = ["BicycleDynamics"]


class BicycleDynamics(Dynamics[VehicleState, VehicleActions, Rectangle]):
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

    def all_actions(self) -> Set[VehicleActions]:
        res = set()
        for acc, dst in product(self.u_acc, self.u_dst):
            res.add(VehicleActions(acc=acc, dst=dst))
        return res

    def successors(self, x: VehicleState, dt: D) -> Mapping[VehicleActions, VehicleState]:
        """ For each state, returns a dictionary U -> Possible Xs """

        # only allow inputs with feasible final vel, st
        u_acc = [_ for _ in self.u_acc if self.v_min <= x.v + _ * dt <= self.v_max]
        u_dst = [_ for _ in self.u_dst if -self.st_max <= x.st + _ * dt <= self.st_max]

        if not u_acc:
            raise ValueError("No feasible acceleration")
        if not u_dst:
            raise ValueError("No feasible steering rate")
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

        alpha = D("1")
        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * (dt * alpha), u_clip)
        ret = x0 + k1 * (dt * (D("0.5") / alpha)) + k2 * (dt * (D("0.5") / alpha))
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

    def get_shared_resources(self, x: VehicleState) -> FrozenSet[Rectangle]:
        # TODO[SIR]: Rectangle assumes heading is along one axis,
        #  change this to generalise for any random heading
        center = x.x, x.y
        sides = self.vg.w, 2 * self.vg.l
        rect = make_rectangle(center=center, sides=sides)
        return frozenset({rect})
