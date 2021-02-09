from typing import Mapping, FrozenSet
import itertools
from decimal import Decimal as D, localcontext
from functools import lru_cache
from frozendict import frozendict
from zuper_commons.types import ZValueError

from games.game_def import Dynamics
from possibilities import Poss, PossibilityMonad

from driving_games.vehicle_dynamics import InvalidAction
from driving_games.structures import SE2_disc, LightsValue, Lights
from duckie_games.shared_resources import get_resources_used, ResourceID, DrivingGameGridMap

from duckie_games.structures import DuckieState, DuckieActions, DuckieGeometry


class DuckieDynamics(Dynamics[DuckieState, DuckieActions, ResourceID]):
    # todo check units
    max_speed: D
    """ Maximum speed [m/s] """

    min_speed: D
    """ Minimum speed [m/s] """

    max_path: D
    """ Maximum distance until end of lane [m] """

    available_accels: FrozenSet[D]
    """ Available acceleration values. """

    max_wait: D
    """ Maximum wait [s] -- maximum duration at v=0. """

    lights_commands: FrozenSet[Lights]
    """ Allowed light commands """

    shared_resources_ds: D
    """ Size of the spatial cells to consider as resources [m]"""

    vg: DuckieGeometry
    """ The duckiebot's geometry. """

    driving_game_grid_map: DrivingGameGridMap
    """ The Driving Game Map containing a resource grid """

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
            vg: DuckieGeometry,
            poss_monad: PossibilityMonad,
            driving_game_grid_map: DrivingGameGridMap
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
        self.driving_game_grid_map = driving_game_grid_map

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[DuckieActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(DuckieActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: DuckieState, dt: D) -> Mapping[DuckieActions, Poss[DuckieState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        possible = {}
        for light, accel in itertools.product(self.lights_commands, self.available_accels):
            u = DuckieActions(accel=accel, light=light)
            try:
                x2 = self.successor(x, u, dt)
            except InvalidAction:
                pass
            else:
                possible[u] = self.ps.unit(x2)

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: DuckieState, u: DuckieActions, dt: D):
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
            # if x2 > self.max_path:
            #     msg = "Invalid action gives out of bound"
            #     raise InvalidAction(msg, x=x, u=u, v2=v2, max_speed=self.max_speed)
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
        ret = DuckieState(
            ref=x.ref,
            lane=x.lane,
            x=x2,
            v=v2,
            wait=wait2,
            light=u.light
        )
        if ret.x < 0:
            raise ZValueError(x=x, u=u, accel_effective=accel_effective, ret=ret)
        return ret

    def get_shared_resources(self, x: DuckieState) -> FrozenSet[ResourceID]:
        return get_resources_used(vs=x, vg=self.vg, m=self.driving_game_grid_map)
