from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from functools import lru_cache, cached_property
from itertools import product
from typing import FrozenSet, Mapping, Set

from frozendict import frozendict
from shapely.geometry import Polygon
from zuper_commons.types import ZException, ZValueError

from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import LightsCmd, LightsValues, NO_LIGHTS
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import Dynamics
from possibilities import Poss, PossibilityMonad
from .resources import get_poly_occupancy
from .resources_occupancy import ResourcesOccupancy, CellIdx
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

    def __post_init__(self):
        if not self.min_speed >= 0:
            msg = "Only forward driving is supported"
            raise ZValueError(msg, min_speed=self.min_speed)


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
        goal_progress: D,
    ):
        self.ref = ref
        self.vg = vg
        self.ps = poss_monad
        self.param = param
        self.min_safety_distance = min_safety_distance
        self.resources_occupancy = resources_occupancy
        self.goal_progress = goal_progress

    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in product(LightsValues, self.param.available_accels):
            res.add(VehicleActions(acc=accel, light=light))
        return frozenset(res)

    @lru_cache(maxsize=2048)
    def successors(self, x: VehicleTrackState, dt: D) -> Mapping[VehicleActions, Poss[VehicleTrackState]]:
        """For each state, returns a dictionary U -> Possible Xs"""
        if x.has_collided:
            # special case for the collided ones, they are forced to stop
            acc_min = min(self.param.available_accels)
            accels = [acc_min if acc_min * dt + x.v >= 0 else D(0)]
            # only one action available for a collided vehicle (either slow down or stay put)
            assert len(accels) == 1, accels
        else:
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
    def successor(self, x: VehicleTrackState, u: VehicleActions, dt: D) -> VehicleTrackState:
        with localcontext() as ctx:
            ctx.prec = 3
            v2 = x.v + u.acc * dt
            if not (self.param.min_speed <= v2 <= self.param.max_speed):
                msg = "Invalid action gives speed out of bounds"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.param.max_speed)
            # only forward moving
            x2 = x.x + (x.v + D("0.5") * u.acc * dt) * dt
            if x2 < x.x:
                raise ZValueError(
                    x=x,
                    x2=x2,
                    u=u,
                    acc=u.acc,
                )
        if v2 == 0:
            wait2 = x.wait + dt
            # who has collided can wait there unlimited time
            if wait2 > self.param.max_wait and not x.has_collided:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = D(0)
        ret = VehicleTrackState(x=x2, v=v2, wait=wait2, light=u.light, has_collided=x.has_collided)

        return ret

    @lru_cache(None)
    def get_shared_resources(self, x: VehicleTrackState, dt: D) -> FrozenSet[CellIdx]:
        max_acc_cmds = self._get_max_acc_commands()
        max_future_x = self.successor(x, max_acc_cmds, dt)
        max_progress = min(max_future_x.x + self.min_safety_distance / 2, self.goal_progress)
        min_progress = x.x  # max(D(0), x.x- self.min_safety_distance/2)
        n_samples = (max_progress - min_progress) // self.occupancy_length
        res: Set[CellIdx] = set()
        for i in range(int(n_samples) + 1):
            x2 = replace(x, x=min_progress + self.occupancy_length * i)
            poly = get_poly_occupancy(vs=x2, vg=self.vg, ref=self.ref)
            ids = self.resources_occupancy.strtree.query_items(poly)
            ids = filter(lambda j: poly.intersects(self.resources_occupancy.strtree._geoms[j]), ids)
            res.update(ids)
        return frozenset(res)

    def _get_max_acc_commands(self) -> VehicleActions:
        return VehicleActions(acc=max(self.param.available_accels), light=NO_LIGHTS)

    def _get_min_acc_commands(self) -> VehicleActions:
        return VehicleActions(acc=min(self.param.available_accels), light=NO_LIGHTS)

    @cached_property
    def occupancy_length(self) -> D:
        with localcontext() as ctx:
            ctx.prec = 3
            res = self.resources_occupancy.cell_resolution + D(self.vg.length + sum(self.vg.bumpers_length))
        return res
