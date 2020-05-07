from dataclasses import dataclass, replace
from functools import lru_cache
from typing import NewType, AbstractSet, FrozenSet, Mapping, Union, Optional
from decimal import Decimal as D

from frozendict import frozendict
from zuper_commons.types import ZValueError, check_isinstance

from driving_games.structures import InvalidAction
from games import Dynamics, PlayerName, Observations, PersonalRewardStructure
from possibilities import One, Poss, ProbabilitySet
from preferences import Preference, SmallerPreferredTol, SmallerPreferred

Pull = NewType("Pull", str)
UP = Pull("up")
DOWN = Pull("down")
PullValue: AbstractSet[Pull] = frozenset(
    {UP, DOWN}
)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    z: Optional[int]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdActions:
    pull: Pull


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdState(object):
    # flying altitude
    z: int = 0
    # augment the state with the stage number for stage-dependent costs
    stage: int = 0


class FlyingDynamics(Dynamics[One, BirdState, BirdActions]):
    """Pulling UP increases x, DOWN decreases"""

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[BirdActions]:
        res = set()
        for pull in PullValue:
            res.add(BirdActions(pull))
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
        if u == UP:
            return replace(x, z=x.z+1, stage=x.stage+1)
        if u == DOWN:
            return replace(x, z=x.z-1, stage=x.stage+1)
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

    def __add__(self, other):
        return BirdCosts(self.cost+other.cost)


class BirdPersonalRewardStructureCustom(
    PersonalRewardStructure[BirdState, BirdActions, BirdCosts]):
    max_stages: int

    def __init__(self, max_stages: int):
        self.max_stages = max_stages

    def personal_reward_incremental(self, x: BirdState, u: BirdActions, dt: D) -> BirdCosts:
        check_isinstance(x, BirdState)
        check_isinstance(u, BirdActions)
        if x.stage == self.max_stages:
            # todo here we should implement the arbitrary payoff matrices
            pass
        else:
            return BirdCosts(1)

    def personal_reward_reduce(self, r1: BirdCosts, r2: BirdCosts) -> BirdCosts:
        return r1+r2

    def personal_final_reward(self, x: BirdState) -> BirdCosts:
        check_isinstance(x, BirdState)
        # assert self.is_personal_final_state(x)
        return BirdCosts(0)

    def is_personal_final_state(self, x: BirdState) -> bool:
        check_isinstance(x, BirdState)
        return x.stage > self.max_stages


class BirdPreferences(SmallerPreferred):
    ...

