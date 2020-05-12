from dataclasses import dataclass, replace
from functools import lru_cache
from typing import NewType, AbstractSet, FrozenSet, Mapping, Union, Optional
from decimal import Decimal as D

from frozendict import frozendict
from zuper_commons.types import ZValueError

from driving_games.structures import InvalidAction
from games import Dynamics, PlayerName, Observations
from possibilities import One, Poss, ProbabilitySet

Go = NewType("Go", str)
UP = Go("up")
DOWN = Go("down")
GoValue: AbstractSet[Go] = frozenset(
    {UP, DOWN}
)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    z: Optional[D]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdActions:
    go: Go


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdState(object):
    # flying altitude
    z: D = 0
    # augment the state with the stage number for stage-dependent costs
    stage: int = 0


class FlyingDynamics(Dynamics[One, BirdState, BirdActions]):
    """Pulling UP increases x, DOWN decreases"""

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[BirdActions]:
        res = set()
        for go in GoValue:
            res.add(BirdActions(go))
        return frozenset(res)

    @lru_cache(None)
    def successors(self, x: BirdState, dt: D) -> Mapping[
        BirdActions, Poss[BirdState, One]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        ps = ProbabilitySet()
        possible = {}
        for u in self.all_actions():
            try:
                x2 = self.successor(x, u)
            except InvalidAction:
                pass
            else:
                possible[u] = ps.lift_one(x2)

        return frozendict(possible)

    @lru_cache(None)
    def successor(self, x: BirdState, u: BirdActions) -> BirdState:
        # trick to get unique NOT path dependent final states and
        # allow arbitrary payoff matrices
        altitude_incr: D = D(1) if x.stage == 0 else D(0.25)
        if u.go == UP:
            return replace(x, z=x.z+altitude_incr, stage=x.stage+1)
        if u.go == DOWN:
            return replace(x, z=x.z-altitude_incr, stage=x.stage+1)
        else:
            raise ZValueError(x=x, u=u)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdObservation:
    others: Mapping[PlayerName, Union[Seen, NotSeen]]


class BirdDirectObservations(Observations[One, BirdState, BirdObservation]):
    possible_states: Mapping[PlayerName, FrozenSet[BirdState]]
    my_possible_states: FrozenSet[BirdState]

    def __init__(
            self,
            my_possible_states: FrozenSet[BirdState],
            possible_states: Mapping[PlayerName, FrozenSet[BirdState]],
    ):
        self.possible_states = possible_states
        self.my_possible_states = my_possible_states

    @lru_cache(None)
    def all_observations(self) -> FrozenSet[BirdObservation]:
        """ Returns all possible observations. """
        assert len(self.possible_states) == 1
        all_of_them = set()
        for me in self.my_possible_states:
            for k, ks_possible_states in self.possible_states.items():
                for ks_possible_state in ks_possible_states:
                    others = {k: ks_possible_state}
                    possible_ys: FrozenSet[BirdObservation] = self.get_observations(me, others)
                    for poss_obs in possible_ys:
                        all_of_them.add(poss_obs)
        return frozenset(all_of_them)

    @lru_cache(None)
    def get_observations(
            self, me: BirdState, others: Mapping[PlayerName, BirdState]
    ) -> FrozenSet[BirdObservation]:
        # ''' For each state, get all possible observations '''
        others = {}
        # fixme this is an empty foo
        for k, v in others.items():
            others[k] = Seen(z=v.z)
        return frozenset({BirdObservation(others)})


@dataclass(frozen=True)
class BirdCosts:
    cost: int
