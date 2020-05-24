import itertools
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from functools import lru_cache
from typing import AbstractSet, FrozenSet, Mapping, NewType, Tuple

from frozendict import frozendict

from games import Dynamics
from possibilities import One, Poss, ProbabilitySet
from zuper_commons.types import ZException, ZValueError
from .rectangle import Rectangle

__all__ = [
    "Lights",
    "NO_LIGHTS",
    "LightsValue",
    "LIGHTS_HEADLIGHTS",
    "LIGHTS_TURN_LEFT",
    "LIGHTS_TURN_RIGHT",
    "VehicleCosts",
    "VehicleGeometry",
]

Lights = NewType("Lights", str)
NO_LIGHTS = Lights("none")
LIGHTS_HEADLIGHTS = Lights("headlights")
LIGHTS_TURN_LEFT = Lights("turn_left")
LIGHTS_TURN_RIGHT = Lights("turn_right")
LightsValue: AbstractSet[Lights] = frozenset(
    {NO_LIGHTS, LIGHTS_HEADLIGHTS, LIGHTS_TURN_LEFT, LIGHTS_TURN_RIGHT}
)

SE2_disc = Tuple[D, D, D]  # in degrees


class InvalidAction(ZException):
    pass


@dataclass(frozen=True)
class VehicleCosts:
    """ The incremental costs"""

    duration: D


@dataclass(frozen=True)
class VehicleGeometry:
    mass: D
    width: D
    length: D
    color: Tuple[float, float, float]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleState:
    # reference frame from where the vehicle started
    ref: SE2_disc
    x: D
    v: D
    # How long we have been at speed = 0. We want to keep track so bound this.
    wait: D
    light: Lights

    __print_order__ = ["x", "v", "wait"]  # only print these attributes


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    accel: D
    light: Lights = "none"


class VehicleDynamics(Dynamics[One, VehicleState, VehicleActions, Rectangle]):
    max_speed: D
    min_speed: D
    max_path: D
    available_accels: FrozenSet[D]
    max_wait: D
    lights_commands: FrozenSet[Lights]
    shared_resources_ds: D
    vg: VehicleGeometry

    def __init__(
        self,
        max_speed: D,
        min_speed: D,
        available_accels: FrozenSet[D],
        max_wait: D,
        ref: SE2_disc,
        max_path: D,
        lights_commands: FrozenSet[Lights],
        shared_resources_ds: D,
        vg: VehicleGeometry,
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.available_accels = available_accels
        self.max_wait = max_wait
        self.ref = ref
        self.max_path = max_path
        self.lights_commands = lights_commands
        self.shared_resources_ds = shared_resources_ds
        self.vg = vg

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: VehicleState, dt: D) -> Mapping[VehicleActions, Poss[VehicleState, One]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accellerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))
        ps = ProbabilitySet()

        possible = {}
        for light, accel in itertools.product(self.lights_commands, self.available_accels):
            u = VehicleActions(accel=accel, light=light)
            try:
                x2 = self.successor(x, u, dt)
            except InvalidAction:
                pass
            else:
                possible[u] = ps.lift_one(x2)

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: VehicleState, u: VehicleActions, dt: D):
        with localcontext() as ctx:
            ctx.prec = 2
            accel_effective = max(-x.v / dt, u.accel)
            v2 = x.v + accel_effective * dt
            if v2 < 0:
                v2 = 0
                # msg = 'Invalid action gives negative vel'
                # raise InvalidAction(msg, x=x, u=u)
            # if v2 < self.min_speed:
            #     v2 = self.min_speed
            if v2 > self.max_speed:
                v2 = self.max_speed
            if not (self.min_speed <= v2 <= self.max_speed):
                msg = "Invalid action gives speed too fast"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
            assert v2 >= 0
            x2 = x.x + (x.v + accel_effective * dt) * dt
            if x2 > self.max_path:
                msg = "Invalid action gives out of bound"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
        # if wait2 > self.max_wait:
        #     msg = f'Invalid action gives wait of {wait2}'
        #     raise InvalidAction(msg, x=x, u=u)

        if v2 == 0:
            wait2 = x.wait + dt
            if wait2 > self.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = D(0)
        ret = VehicleState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light)
        if ret.x < 0:
            raise ZValueError(x=x, u=u, accel_effective=accel_effective, ret=ret)
        return ret

    def get_shared_resources(self, x: VehicleState) -> FrozenSet[Rectangle]:
        from .collisions_check import get_resources_used

        return get_resources_used(vs=x, vg=self.vg, ds=self.shared_resources_ds)

    # @lru_cache(None)
    # def assert_valid_state(self, s: VehicleState):
    #     if s.wait and s.v:
    #         raise ZValueError(s=s)
    #
    #     if not (0 <= s.x <= self.max_path):
    #         raise ZValueError(s=s)
    #     if not (0 <= s.v <= self.max_speed):
    #         raise ZValueError(s=s)
    #     if not (0 <= s.wait <= self.max_wait):
    #         raise ZValueError(s=s)
