from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import NewType, AbstractSet, FrozenSet, Mapping, Union, Optional, Tuple, List
from decimal import Decimal as D
import numpy as np
from matplotlib import pyplot
from matplotlib import patches

from frozendict import frozendict
from zuper_commons.types import ZValueError

from driving_games.vehicle_dynamics import InvalidAction
from games import Dynamics, PlayerName, Observations, GameVisualization
from possibilities import Poss, PossibilityMonad

Step = NewType("Step", str)
"""Toy Cars"""
WAIT = Step("+0")
PLUSONE = Step("+1")
PLUSTWO = Step("+2")
AvailableSteps: AbstractSet[Step] = frozenset({
    WAIT,
    PLUSONE,
    # PLUSTWO
})


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

    lane: ToyLane
    # Spacetime
    x: CtrPointID  # corresponds to along the lane (x=0 first ctr. point, x=1 second ctr. point)
    # time: int
    wait: int

    @property
    def point_in_map(self) -> PointInMap:
        return self.lane.from_along_lane_to_point_in_map(along_lane=self.x)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    state: ToyCarState


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class ToyResources:
    # time: int
    point_in_map: int


class ToyCarDynamics(Dynamics[ToyCarState, ToyCarActions, ToyResources]):
    """Step of +1 or +2 or wait"""

    def __init__(self, poss_monad: PossibilityMonad, max_path: CtrPointID, max_wait: int):
        self.ps = poss_monad
        self.max_path = max_path
        self.max_wait = max_wait

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[ToyCarActions]:
        res = set()
        for step in AvailableSteps:
            res.add(ToyCarActions(step))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: ToyCarState, dt: D) -> Mapping[ToyCarActions, Poss[ToyCarState]]:
        """ For each state, returns the possible outcomes given certain actions """
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

        if u.step == PLUSONE:
            along_lane = x.x + 1

            if along_lane > self.max_path:
                along_lane = self.max_path
                # msg = "Invalid action gives out of bounds"
                # raise InvalidAction(msg, x=x, u=u, along_lane=along_lane, max_path=self.max_path)

            # xnew = replace(x, x=along_lane, time=x.time + 1, wait=0)
            xnew = replace(x, x=along_lane, wait=0)

        elif u.step == PLUSTWO:
            along_lane = x.x + 2
            if along_lane > self.max_path:
                along_lane = self.max_path
                # msg = "Invalid action gives out of bounds"
                # raise InvalidAction(msg, x=x, u=u, along_lane=along_lane, max_path=self.max_path)

            # xnew = replace(x, x=along_lane, time=x.time + 1, wait=0)
            xnew = replace(x, x=along_lane, wait=0)

        elif u.step == WAIT:
            wait2 = x.wait + 1
            if wait2 > self.max_wait:
                msg = f"Invalid action gives wait of {wait2}"
                raise InvalidAction(msg, x=x, u=u)
            else:
                # xnew = replace(x, time=x.time + 1, wait=wait2)
                xnew = replace(x, wait=wait2)

        else:
            raise ZValueError(x=x, u=u)

        return xnew

    def get_shared_resources(self, x: ToyCarState) -> FrozenSet[ToyResources]:
        # resources = [ToyResources(time=x.time, point_in_map=x.point_in_map)]
        resources = [ToyResources(point_in_map=x.point_in_map)]
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
        self.lanes = self.toy_map.lanes
        self.lane_y_pos = np.linspace(0, 1, len(self.lanes) + 2)[1:-1]
        self.player_color = [(y_pos, 0, 1) for y_pos in self.lane_y_pos]
        self.pylab: pyplot = None

    def hint_graph_node_pos(self, state: ToyCarState) -> Tuple[float, float]:
        # return float(state.x), float(state.time)
        return float(state.x), float(state.lane.control_points[0])

    def plot_player(
            self,
            player_name: PlayerName,
            state: ToyCarState,
            commands: Optional[ToyCarActions],
            opacity: float = 1.0
    ):
        for i, lane in enumerate(self.lanes):
            if lane is state.lane:
                along_lane = state.x
                y = self.lane_y_pos[i]
                self.pylab.text(
                    along_lane,
                    y,
                    player_name,
                    zorder=15,
                    horizontalalignment="center",
                    verticalalignment="top",
                )
                toy_car = patches.Circle((along_lane, y), radius=0.05, color=self.player_color[i])
                ax = self.pylab.gca()
                ax.add_patch(toy_car)
                break

    @contextmanager
    def plot_arena(self, pylab: pyplot, ax):
        self.pylab = pylab
        ctr_pts = [lane.control_points for lane in self.lanes]
        max_lane_length = max(map(len, ctr_pts))
        ax.set_xlim(left=-0.2, right=max_lane_length - 0.8)
        ax.set_ylim(bottom=0, top=1)
        for i, ctr_p in enumerate(ctr_pts):
            x_lane = np.array(list(ctr_p.keys()))
            y_lane = np.array([self.lane_y_pos[i]] * len(x_lane))
            pylab.plot(x_lane, y_lane)
            for j, ctr_p_value in ctr_p.items():
                pylab.text(
                    j,
                    self.lane_y_pos[i],
                    str(ctr_p_value),
                    zorder=15,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        yield

        pylab.axis("off")
        ax.set_aspect("equal")
