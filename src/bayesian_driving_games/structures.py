import itertools
from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from functools import lru_cache
from typing import FrozenSet, Mapping, NewType, List

from frozendict import frozendict

from driving_games import VehicleActions
from driving_games.structures import InvalidAction, LightsValue, VehicleState, VehicleDynamics
from games import Dynamics, GamePlayer
from possibilities import Poss
from zuper_commons.types import ZValueError
from driving_games.rectangle import Rectangle

__all__ = ["PlayerType", "BayesianVehicleState", "BayesianVehicleDynamics"]

PlayerType = NewType("PlayerType", str)
""" The type of a player. """

IN_A_RUSH = PlayerType("in a rush")
""" rushed player type optimizes for time. """

RELAXED = PlayerType("relaxed")
""" relaxed player optimizes for comfort """

NO_TYPE = PlayerType("no type")
""" No types assigned """


@dataclass
class BayesianGamePlayer(GamePlayer):
    types_of_other: List[PlayerType]
    """The types of the other player""" # TODO: Extend to more than two players - Mapping[PlayerName, List[PlayerType]]

    types_of_myself: List[PlayerType]
    """The types of myself"""

    prior: Poss[PlayerType]
    """ The prior over the other player's types"""


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianVehicleState(VehicleState):
    player_type: PlayerType
    """ The type of the player """

    __print_order__ = ["x", "v", "player_type"]  # only print these attributes

    def compare_physical_states(self, s2: "BayesianVehicleState") -> bool:
        """ returns True if they are in the same VehicleState, False otherwise """
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


class BayesianVehicleDynamics(VehicleDynamics, Dynamics[BayesianVehicleState, VehicleActions, Rectangle]):
    def __init__(self, player_types: FrozenSet[PlayerType], **kwargs):
        super().__init__(**kwargs)
        self.player_types = player_types

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[VehicleActions]:
        res = set()
        for light, accel in itertools.product(LightsValue, self.available_accels):
            res.add(VehicleActions(accel=accel, light=light))
        return frozenset(res)

    @lru_cache(None)
    def successors(
        self, x: BayesianVehicleState, dt: D
    ) -> Mapping[VehicleActions, Poss[BayesianVehicleState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # only allow accelerations that make the speed non-negative
        accels = [_ for _ in self.available_accels if _ * dt + x.v >= 0]
        # if the speed is 0 make sure we cannot wait forever
        if x.wait >= self.max_wait:
            assert x.v == 0, x
            accels.remove(D(0))

        if x.player_type == NO_TYPE:
            possible = {}
            u = VehicleActions(accel=None, light=None)  # TODO: fixme accel=None does not look good
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
        ret = BayesianVehicleState(
            ref=x.ref, x=x2, v=v2, wait=wait2, light=u.light, player_type=x.player_type
        )
        if ret.x < 0:
            raise ZValueError(x=x, u=u, accel_effective=accel_effective, ret=ret)
        return ret

    def get_shared_resources(self, x: BayesianVehicleState) -> FrozenSet[Rectangle]:
        from driving_games.collisions_check import get_resources_used

        return get_resources_used(vs=x, vg=self.vg, ds=self.shared_resources_ds)
