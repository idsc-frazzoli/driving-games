from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from functools import lru_cache
from itertools import product
from typing import Mapping, FrozenSet

from frozendict import frozendict
from shapely.geometry import Polygon
from zuper_commons.types import ZValueError, ZException

from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import LightsValues, LightsCmd
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import Dynamics
from possibilities import Poss, PossibilityMonad
from .structures import (
    VehicleTrackState,
    VehicleActions,
)

__all__ = ["VehicleTrackDynamicsParams", "VehicleTrackDynamics", "InvalidAction"]


class InvalidAction(ZException):
    pass


@dataclass(frozen=True)
class VehicleTrackDynamicsParams:
    max_speed: D
    """ Maximum speed [m/s] """
    min_speed: D
    """ Minimum speed [m/s] """
    available_accels: FrozenSet[D]
    """ Available acceleration values. """
    max_wait: D
    """ Maximum wait [s] -- maximum duration at v=0. """
    lights_commands: FrozenSet[LightsCmd]
    """ Allowed light commands """
    shared_resources_ds: D
    """ Size of the spatial cells to consider as resources [m]"""


class VehicleTrackDynamics(Dynamics[VehicleTrackState, VehicleActions, Polygon]):
    """Dynamics only along a DGLanelet"""

    max_path: D
    """ Maximum `s` until end of episode [m] """
    vg: VehicleGeometry
    """ The vehicle's geometry. """
    param: VehicleTrackDynamicsParams

    def __init__(
        self,
        ref: DgLanelet,
        max_path: D,
        vg: VehicleGeometry,
        poss_monad: PossibilityMonad,
        param: VehicleTrackDynamicsParams,
    ):
        self.ref = ref
        self.max_path = max_path
        self.vg = vg
        self.ps = poss_monad
        self.param = param

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in product(LightsValues, self.param.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: VehicleTrackState, dt: D) -> Mapping[VehicleActions, Poss[VehicleTrackState]]:
        """For each state, returns a dictionary U -> Possible Xs"""
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.param.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.param.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        possible = {}
        for light, accel in product(self.param.lights_commands, self.param.available_accels):
            u = VehicleActions(accel=accel, light=light)
            try:
                x2 = self.successor(x, u, dt)
            except InvalidAction:
                pass
            else:
                possible[u] = self.ps.unit(x2)

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: VehicleTrackState, u: VehicleActions, dt: D):
        with localcontext() as ctx:
            ctx.prec = 2
            accel_effective = max(-x.v / dt, u.acc)
            v2 = x.v + accel_effective * dt
            if v2 < 0:
                v2 = 0
                # msg = 'Invalid action gives negative vel'
                # raise InvalidAction(msg, x=x, u=u)
            # if v2 < self.min_speed:
            #     v2 = self.min_speed
            if v2 > self.param.max_speed:
                v2 = self.param.max_speed
            if not (self.param.min_speed <= v2 <= self.param.max_speed):
                msg = "Invalid action gives speed too fast"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.param.max_speed)
            assert v2 >= 0
            x2 = x.x + (x.v + accel_effective * dt) * dt
            if x2 > self.max_path:
                msg = "Invalid action gives out of bound"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.param.max_speed)
        # if wait2 > self.max_wait:
        #     msg = f'Invalid action gives wait of {wait2}'
        #     raise InvalidAction(msg, x=x, u=u)

        if v2 == 0:
            wait2 = x.wait + dt
            if wait2 > self.param.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = D(0)
        ret = VehicleTrackState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light)
        if ret.x < 0:
            raise ZValueError(x=x, u=u, accel_effective=accel_effective, ret=ret)
        return ret

    def get_shared_resources(self, x: VehicleTrackState) -> FrozenSet[Polygon]:
        return get_resources_used(vs=x, vg=self.vg, ds=self.param.shared_resources_ds)
