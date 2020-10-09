import itertools
from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import AbstractSet, FrozenSet, Mapping, NewType, Tuple

from frozendict import frozendict

from games import Dynamics
from possibilities import Poss, PossibilitySet, PossibilityMonad
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
    "VehicleState",
    "VehicleActions",
    "VehicleGeometry",
]

Lights = NewType("Lights", str)
""" The type of light commands. """

NO_LIGHTS = Lights("none")
""" Lights are off. """
LIGHTS_HEADLIGHTS = Lights("headlights")
""" The front lights are on. """
LIGHTS_TURN_LEFT = Lights("turn_left")
""" Blinkers turn left """
LIGHTS_TURN_RIGHT = Lights("turn_right")
""" Blinkers turn right """
LightsValue: AbstractSet[Lights] = frozenset(
    {NO_LIGHTS, LIGHTS_HEADLIGHTS, LIGHTS_TURN_LEFT, LIGHTS_TURN_RIGHT}
)
""" All possible lights command value"""

SE2_disc = Tuple[D, D, D]  # in degrees


class InvalidAction(ZException):
    pass


@dataclass(frozen=True)
class VehicleCosts:
    """ The personal costs of the vehicle"""

    duration: D
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleCosts":
        # weighting costs, e.g. according to a probability
        return replace(self, duration=self.duration * D(float(weight)))

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleCosts") -> "VehicleCosts":
        if type(other) == VehicleCosts:
            return replace(self, duration=self.duration + other.duration)
        else:
            if other is None:
                return self
            else:
                raise NotImplementedError

    __radd__ = __add__


@dataclass(frozen=True)
class VehicleGeometry:
    mass: D
    """ Mass [kg] """
    width: D
    """ Car width [m] """
    length: D
    """ Car length [m] """
    color: Tuple[float, float, float]
    """ Car color """


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleState:
    ref: SE2_disc
    """ Reference frame from where the vehicle started """

    x: D
    """ Longitudinal position """

    v: D
    """ Longitudinal velocity """

    wait: D
    """ How long we have been at speed = 0. We want to keep track so bound this. """

    light: Lights
    """ The current lights signal. """

    __print_order__ = ["x", "v"]  # only print these attributes


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    accel: D
    light: Lights = NO_LIGHTS


class VehicleDynamics(Dynamics[VehicleState, VehicleActions, Rectangle]):
    max_speed: D
    """ Maximum speed [m/s] """

    min_speed: D
    """ Minimum speed [m/s] """

    max_path: D
    """ Maximum `x` until end of episode [m] """

    available_accels: FrozenSet[D]
    """ Available accelleration values. """

    max_wait: D
    """ Maximum wait [s] -- maximum duration at v=0. """

    lights_commands: FrozenSet[Lights]
    """ Allowed light commands """

    shared_resources_ds: D
    """ Size of the spatial cells to consider as resources [m]"""

    vg: VehicleGeometry
    """ The vehicle's geometry. """

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
        poss_monad: PossibilityMonad,
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
        self.ps = poss_monad

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: VehicleState, dt: D) -> Mapping[VehicleActions, Poss[VehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        possible = {}
        for light, accel in itertools.product(self.lights_commands, self.available_accels):
            u = VehicleActions(accel=accel, light=light)
            try:
                x2 = self.successor(x, u, dt)
            except InvalidAction:
                pass
            else:
                possible[u] = self.ps.unit(x2)

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
