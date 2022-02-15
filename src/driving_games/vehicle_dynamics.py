from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from functools import lru_cache
from itertools import product
from typing import FrozenSet, Mapping, Set

from frozendict import frozendict
from shapely.geometry import Polygon

from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import LightsCmd, LightsValues, NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import Dynamics
from possibilities import Poss, PossibilityMonad
from zuper_commons.types import ZException, ZValueError
from .resources import get_poly_occupancy, PolygonHashable
from .resources_occupancy import ResourcesOccupancy, CellID
from .structures import VehicleActions, VehicleTrackState

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
    shared_resources_ds: float
    """ Size of the spatial cells to consider as resources [m]"""


class VehicleTrackDynamics(Dynamics[VehicleTrackState, VehicleActions, Polygon]):
    """Dynamics only along a DGLanelet"""

    vg: VehicleGeometry
    """ The vehicle's geometry. """
    param: VehicleTrackDynamicsParams
    """ The parameters for the dynamics. """
    min_safety_distance: D
    """ The minimum safety distance. Used to construct shared resources that bound the joint costs. """
    resources_occupancy: ResourcesOccupancy
    """ A structure to extract the occupied resources"""

    def __init__(
        self,
        ref: DgLanelet,
        vg: VehicleGeometry,
        poss_monad: PossibilityMonad,
        param: VehicleTrackDynamicsParams,
        min_safety_distance: D,
        resources_occupancy: ResourcesOccupancy,
    ):
        self.ref = ref
        self.vg = vg
        self.ps = poss_monad
        self.param = param
        self.min_safety_distance = min_safety_distance
        self.resources_occupancy = resources_occupancy

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in product(LightsValues, self.param.available_accels):
            res.add(VehicleActions(acc=accel, light=light))
        return frozenset(res)

    @lru_cache(maxsize=2048)
    def successors(self, x: VehicleTrackState, dt: D) -> Mapping[VehicleActions, Poss[VehicleTrackState]]:
        """For each state, returns a dictionary U -> Possible Xs"""
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.param.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait > self.param.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        possible = {}
        for light, accel in product(self.param.lights_commands, accels):
            u = VehicleActions(acc=accel, light=light)
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
            ctx.prec = 4
            v2 = x.v + u.acc * dt
            if not (self.param.min_speed <= v2 <= self.param.max_speed):
                msg = "Invalid action gives speed out of bounds"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.param.max_speed)
            # only forward moving
            assert v2 >= 0, v2
            x2 = x.x + (x.v + D("0.5") * u.acc * dt) * dt
            if x2 < x.x:
                raise ZValueError(
                    x=x,
                    u=u,
                    acc=u.acc,
                )
        if v2 == 0:
            wait2 = x.wait + dt
            if wait2 > self.param.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = D(0)
        ret = VehicleTrackState(x=x2, v=v2, wait=wait2, light=u.light, has_collided=x.has_collided)

        return ret

    def get_shared_resources(self, x: VehicleTrackState, dt: D) -> FrozenSet[CellID]:
        # todo: this is not correct, we should use the lanelet graph
        max_acc_cmds = self._get_max_acc_commands()
        max_future_x = self.successor(x, max_acc_cmds, dt)
        max_future_x2 = replace(max_future_x, x=max_future_x.x + self.min_safety_distance)
        res: Set[CellID] = set()
        for x in [x, max_future_x, max_future_x2]:
            poly = get_poly_occupancy(vs=x, vg=self.vg, ref=self.ref)
            ids = self.resources_occupancy.strtree.query_items(poly)
            res.update(ids)
        return frozenset(res)

    def _get_max_acc_commands(self) -> VehicleActions:
        return VehicleActions(acc=max(self.param.available_accels), light=NO_LIGHTS)
