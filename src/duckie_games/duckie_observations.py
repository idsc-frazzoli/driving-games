from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Mapping, Optional, Union
from decimal import Decimal as D

from duckietown_world.world_duckietown.lane_segment import LaneSegment

from games import Observations, PlayerName
from driving_games.structures import Lights, SE2_disc
from driving_games.vehicle_observation import NotSeen

from duckie_games.structures import DuckieState


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    ref: SE2_disc
    lane: LaneSegment
    x: Optional[int]
    v: Optional[int]
    # if not None, we could also see the light value
    light: Optional[Lights]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class DuckieDirectObservations(Observations[DuckieState, DuckieObservation]):
    possible_states: Mapping[PlayerName, FrozenSet[DuckieState]]
    my_possible_states: FrozenSet[DuckieState]

    def __init__(
        self,
        my_possible_states: FrozenSet[DuckieState],
        possible_states: Mapping[PlayerName, FrozenSet[DuckieState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> FrozenSet[DuckieObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: FrozenSet[DuckieObservation] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
        self, me: DuckieState, others: Mapping[PlayerName, DuckieState]
    ) -> FrozenSet[DuckieObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(ref=v.ref, lane=v.lane, x=v.x, v=v.v, light=None)
        return frozenset({DuckieObservation(others)})
