import itertools
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from functools import lru_cache
from typing import cast, FrozenSet as ASet, Mapping, Optional, Sequence, Tuple, Union

from frozendict import frozendict
from typing_extensions import Literal
from zuper_commons.types import ZException

from games import Dynamics, Observations, PlayerName

Lights = Literal["none", "headlights", "turn_left", "turn_right"]
# noinspection PyTypeChecker
LightsValue: Sequence[Lights] = ["none", "headlights", "turn_left", "turn_right"]
NO_LIGHTS = cast(Lights, "none")
SE2_disc = Tuple[D, D, D]  # in degrees


class InvalidAction(ZException):
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleState:
    # reference frame from where the vehicle started
    ref: SE2_disc
    x: D
    v: D
    # how long at speed = 0 (we want to bound)
    wait: D
    light: Lights

    __print_order__ = ["x", "v"]  # only print these attributes


@dataclass(frozen=True)
class VehicleActions:
    accel: D
    light: Lights = "none"


class VehicleDynamics(Dynamics[VehicleState, VehicleActions]):
    max_speed: D
    min_speed: D
    max_path: D
    available_accels: ASet[D]
    max_wait: D
    lights_commands: ASet[Lights]

    def __init__(
        self,
        max_speed: D,
        min_speed: D,
        available_accels: ASet[D],
        max_wait: D,
        ref: SE2_disc,
        max_path: D,
        lights_commands: ASet[Lights],
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.available_accels = available_accels
        self.max_wait = max_wait
        self.ref = ref
        self.max_path = max_path
        self.lights_commands = lights_commands

    @lru_cache(None)
    def all_actions(self) -> ASet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: VehicleState, dt: D) -> Mapping[VehicleActions, ASet[VehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accellerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ + x.v >= 0]
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
                possible[u] = frozenset({x2})

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: VehicleState, u: VehicleActions, dt: D):
        with localcontext() as ctx:
            ctx.prec = 2
            v2 = x.v + u.accel * dt
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
            x2 = x.x + (x.v + u.accel * dt) * dt
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
        return VehicleState(ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light)

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


@dataclass
class NotSeen:
    pass


@dataclass
class Seen:
    ref: SE2_disc
    x: Optional[int]
    v: Optional[int]
    # if not None, we could also see the light value
    light: Optional[Lights]


@dataclass
class VehicleObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class VehicleDirectObservations(Observations[VehicleState, VehicleObservation]):
    possible_states: Mapping[PlayerName, ASet[VehicleState]]
    my_possible_states: ASet[VehicleState]

    def __init__(
        self,
        my_possible_states: ASet[VehicleState],
        possible_states: Mapping[PlayerName, ASet[VehicleState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> ASet[VehicleObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: ASet[VehicleObservation] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
        self, me: VehicleState, others: Mapping[PlayerName, VehicleState]
    ) -> ASet[VehicleObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, x=v.x, v=v.v, light=None)
        return frozenset({VehicleObservation(others)})


@dataclass(frozen=True, unsafe_hash=True, order=True)
class CollisionCost:
    v: D
