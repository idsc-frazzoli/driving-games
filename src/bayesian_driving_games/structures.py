import itertools
from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import AbstractSet, FrozenSet, Mapping, NewType, Tuple

from frozendict import frozendict

from driving_games import VehicleGeometry, VehicleActions
from driving_games.structures import InvalidAction, VehicleCosts, Lights, LightsValue
from games import Dynamics, PlayerName
from possibilities import Poss, PossibilitySet, PossibilityMonad
from zuper_commons.types import ZException, ZValueError
from driving_games.rectangle import Rectangle

__all__ = [
    "Lights",
    "NO_LIGHTS",
    "LightsValue",
    "LIGHTS_HEADLIGHTS",
    "LIGHTS_TURN_LEFT",
    "LIGHTS_TURN_RIGHT",
    "VehicleCosts",
    "BayesianVehicleState",
    "VehicleActions",
    "VehicleGeometry",
]

PlayerType = NewType("PlayerType", str)
""" The type of a player. """


SE2_disc = Tuple[D, D, D]  # in degrees



@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianVehicleState:
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

    player_type: PlayerType
    """ The type of the player """

    __print_order__ = ["x", "v", "player_type"]  # only print these attributes

    def compare_physical_states(self, s2) -> bool:
        if self.ref != s2.ref:
            return False
        elif self.x != s2.x:
            return False
        elif self.v != s2.v:
            return False
        elif self.wait != s2.wait:
            return False
        elif self.light != s2.light:
            return False
        elif self.player_type == s2.player_type:
            return False
        else:
            return True


class VehicleDynamics(Dynamics[BayesianVehicleState, VehicleActions, Rectangle]):
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
            player_types: FrozenSet[PlayerType]
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
        self.player_types = player_types

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: BayesianVehicleState, dt: D) -> Mapping[VehicleActions, Poss[BayesianVehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _*dt+x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        if x.player_type==PlayerType('0'):
            possible = {}
            u = VehicleActions(accel=None, light=None)
            for _ in self.player_types:
                x2 = BayesianVehicleState(ref=x.ref, x=x.x, v=x.v, wait=x.wait, light=x.light, player_type=_)
                possible[x2.player_type] = self.ps.unit(x2)

            return frozendict(possible)

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
    def successor(self, x: BayesianVehicleState, u: VehicleActions, dt: D):
        with localcontext() as ctx:
            ctx.prec = 2
            accel_effective = max(-x.v/dt, u.accel)
            v2 = x.v+accel_effective*dt
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
            x2 = x.x+(x.v+accel_effective*dt)*dt
            if x2 > self.max_path:
                msg = "Invalid action gives out of bound"
                raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
        # if wait2 > self.max_wait:
        #     msg = f'Invalid action gives wait of {wait2}'
        #     raise InvalidAction(msg, x=x, u=u)


        if v2 == 0:
            wait2 = x.wait+dt
            if wait2 > self.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
        else:
            wait2 = D(0)
        ret = BayesianVehicleState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light, player_type=x.player_type)
        if ret.x < 0:
            raise ZValueError(x=x, u=u, accel_effective=accel_effective, ret=ret)
        return ret


    def get_shared_resources(self, x: BayesianVehicleState) -> FrozenSet[Rectangle]:
        from driving_games.collisions_check import get_resources_used

        return get_resources_used(vs=x, vg=self.vg, ds=self.shared_resources_ds)
