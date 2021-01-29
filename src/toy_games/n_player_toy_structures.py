from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import NewType, AbstractSet, FrozenSet, Mapping, Union, Optional, Tuple, List, Dict
from decimal import Decimal as D

from frozendict import frozendict
from zuper_commons.types import ZValueError

from driving_games.vehicle_dynamics import InvalidAction
from games import Dynamics, PlayerName, Observations, GameVisualization
from possibilities import Poss, PossibilityMonad

Step = NewType("Step", str)
"""Toy Cars"""
PLUSONE = Step("+1")
PLUSTWO = Step("+2")
AvailableSteps: AbstractSet[Step] = frozenset({PLUSONE, PLUSTWO})


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyCarActions:
    step: Step

PointInMap = int
CtrPointID = int

@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyLane:
    control_points: Mapping[CtrPointID, PointInMap]

    def from_along_lane_to_point_in_map(self, along_lane: CtrPointID) -> PointInMap:
        return self.control_points[along_lane]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyCarMap:
    lanes: List[ToyLane]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyCarState(object):
    __slots__ = ["lane", "along_lane", "time"]
    lane: ToyLane
    # Spacetime
    along_lane: CtrPointID
    time: int

    @property
    def point_in_map(self) -> PointInMap:
        return self.lane.from_along_lane_to_point_in_map(along_lane=self.along_lane)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    state: ToyCarState


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyResources:
    time: int
    point_in_map: int


class ToyCarDynamics(Dynamics[ToyCarState, ToyCarActions, ToyResources]):
    """Step of +1 or +2"""

    def __init__(self, poss_monad: PossibilityMonad, max_path: CtrPointID):
        self.ps = poss_monad
        self.max_path = max_path

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[ToyCarActions]:
        res = set()
        for step in AvailableSteps:
            res.add(ToyCarActions(step))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: ToyCarState, dt: D) -> Mapping[ToyCarActions, Poss[ToyCarState]]:
        """ For each state, returns the possible outcomes given certain actions """
        # todo expand to allow other possibility monads
        possible = {}
        for u in self.all_actions():
            try:
                x2 = self.successor(x, u)
            except InvalidAction:
                pass
            else:
                possible[u] = self.ps.unit(x2)

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: ToyCarState, u: ToyCarActions) -> ToyCarState:
        # trick to get unique NOT path dependent final states and
        # allow arbitrary payoff matrices
        if u.step == PLUSONE:
            along_lane = x.along_lane + 1 if x.along_lane + 1 <= self.max_path else self.max_path

            return replace(x, along_lane=along_lane, time=x.time + 1)

        if u.step == PLUSTWO:

            along_lane = x.along_lane + 2 if x.along_lane + 2 <= self.max_path else self.max_path

            return replace(x, along_lane=along_lane, time=x.time + 1)
        else:
            raise ZValueError(x=x, u=u)

    def get_shared_resources(self, x: ToyCarState) -> FrozenSet[ToyResources]:
        resources = [ToyResources(time=x.time, point_in_map=x.point_in_map)]
        return frozenset(resources)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyCarObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class ToyCarDirectObservations(Observations[ToyCarState, ToyCarObservation]):
    possible_states: Mapping[PlayerName, FrozenSet[ToyCarState]]
    my_possible_states: FrozenSet[ToyCarState]

    def __init__(
        self,
        my_possible_states: FrozenSet[ToyCarState],
        possible_states: Mapping[PlayerName, FrozenSet[ToyCarState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> FrozenSet[ToyCarObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: FrozenSet[ToyCarObservation] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
        self, me: ToyCarState, others: Mapping[PlayerName, ToyCarState]
    ) -> FrozenSet[ToyCarObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        for k, v in others.items():
            others[k] = Seen(state=v)
        return frozenset({ToyCarObservation(others)})


@dataclass(frozen=True)
class ToyCarCosts:

    __slots__ = ["duration"]
    duration: D
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "ToyCarCosts":
        # weighting costs, e.g. according to a probability
        return replace(self, duration=self.duration * D(float(weight)))

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "ToyCarCosts") -> "ToyCarCosts":
        if isinstance(other, ToyCarCosts):
            return replace(self, duration=self.duration + other.duration)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__


class ToyCarVisualization(
    GameVisualization[ToyCarState, ToyCarActions, ToyCarDirectObservations, ToyCarCosts, ToyCarCosts]
):
    def __init__(self, toy_map: ToyCarMap):
        self.toy_map = toy_map


    def hint_graph_node_pos(self, state: ToyCarState) -> Tuple[float, float]:
        pass

    def plot_player(self, player_name: PlayerName, state: ToyCarState, commands: Optional[ToyCarActions], opacity: float = 1.0):
        pass

    @contextmanager
    def plot_arena(self, pylab, ax):
        yield

