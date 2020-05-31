from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Mapping, Optional, Union

from games import Observations, PlayerName
from .structures import Lights, SE2_disc, VehicleState

__all__ = ["NotSeen", "Seen", "VehicleObservation", "VehicleDirectObservations"]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    ref: SE2_disc
    x: Optional[int]
    v: Optional[int]
    # if not None, we could also see the light value
    light: Optional[Lights]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class VehicleDirectObservations(Observations[VehicleState, VehicleObservation]):
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
    def all_observations(self) -> FrozenSet[VehicleObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: FrozenSet[VehicleObservation] = self.get_observations(
                        me, others
                    )
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
        self, me: VehicleState, others: Mapping[PlayerName, VehicleState]
    ) -> FrozenSet[VehicleObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, x=v.x, v=v.v, light=None)
        return frozenset({VehicleObservation(others)})
