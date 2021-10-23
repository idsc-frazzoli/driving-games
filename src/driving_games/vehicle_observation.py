from dataclasses import dataclass
from decimal import Decimal as D
from functools import lru_cache
from typing import FrozenSet, Mapping, Optional, Union

from dg_commons import PlayerName, SE2Transform
from dg_commons.sim.models.vehicle_ligths import LightsCmd
from games import Observations
from .structures import VehicleState

__all__ = ["NotSeen", "Seen", "VehicleObs", "VehicleDirectObservations"]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    ref: SE2Transform
    x: Optional[int]
    v: Optional[int]
    # if not None, we could also see the light value
    light: Optional[LightsCmd]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleObs:
    """Vehicle observations"""

    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class VehicleDirectObservations(Observations[VehicleState, VehicleObs]):
    possible_states: Mapping[PlayerName, FrozenSet[VehicleState]]
    my_possible_states: FrozenSet[VehicleState]

    def __init__(
        self,
        my_possible_states: FrozenSet[VehicleState],
        possible_states: Mapping[PlayerName, FrozenSet[VehicleState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> FrozenSet[VehicleObs]:
        """Returns all possible observations."""
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: FrozenSet[VehicleObs] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(self, me: VehicleState, others: Mapping[PlayerName, VehicleState]) -> FrozenSet[VehicleObs]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, x=v.x, v=v.v, light=None)
        return frozenset({VehicleObs(others)})


class TwoVehicleSeenObservation:
    def __init__(self, road: D, side: D, p1_ref: tuple, p2_ref: tuple):
        self.road = road
        self.side = side
        self.p1_ref = p1_ref
        self.p2_ref = p2_ref

    def do_we_see_us(self, first_progress: D, second_progress: D) -> bool:

        # This function determines if the two cars can see each other. If a straight line between the two cars
        # contains only road, they can see each other, else they can't.

        a1, a2 = self.p1_ref[0], self.p1_ref[1] + first_progress
        b1, b2 = self.p2_ref[0] - second_progress, self.p2_ref[1]
        side_left = self.side
        side_right = self.side + self.road

        t1_left = (side_left - a1) / (b1 - a1)
        t1_right = (side_right - a1) / (b1 - a1)

        if 0 < t1_left < 1:
            res_left = a2 + t1_left * (b2 - a2)
        else:
            res_left = (2 * self.side + self.road) / 2
        if 0 < t1_right < 1:
            res_right = a2 + t1_right * (b2 - a2)
        else:
            res_right = (2 * self.side + self.road) / 2

        if res_left < side_left:
            return False
        elif res_left > side_right:
            return False
        elif res_right < side_left:
            return False
        elif res_right > side_right:
            return False
        else:
            return True
