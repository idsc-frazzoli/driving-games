from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import NewType, AbstractSet, FrozenSet, Mapping, Union, Optional, Tuple, List
from decimal import Decimal as D

from frozendict import frozendict
from zuper_commons.types import ZValueError

from driving_games.structures import InvalidAction
from games import Dynamics, PlayerName, Observations, X, GameVisualization, U
from games.game_def import SR
from possibilities import Poss, PossibilitySet, PossibilityMonad

Go = NewType("Go", str)
UP = Go("up")
DOWN = Go("down")
GoValue: AbstractSet[Go] = frozenset({UP, DOWN})


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class NotSeen:
    pass


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class Seen:
    z: Optional[D]


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianBirdActions:
    go: Go


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BirdState(object):
    # flying altitude
    z: D = 0
    # augment the state with the stage number for stage-dependent costs
    stage: int = 0


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class BayesianBirdState(BirdState):
    # type
    player_type: str = "0"

    def compare_physical_states(self, s2) -> bool:
        if self.z != s2.z:
            return False
        elif self.stage != s2.stage:
            return False
        elif self.player_type == s2.player_type:
            return False
        else:
            return True


class BayesianFlyingDynamics(Dynamics[BayesianBirdState, BayesianBirdActions, SR]):
    """Pulling UP increases x, DOWN decreases"""

    def __init__(self, poss_monad: PossibilityMonad, types: List):
        self.ps = poss_monad
        self.types = types

    @lru_cache(None)
    def all_actions(self) -> FrozenSet[BayesianBirdActions]:
        res = set()
        for go in GoValue:
            res.add(BayesianBirdActions(go))
        return frozenset(res)

    @lru_cache(None)
    def successors(
        self, x: BayesianBirdState, dt: D
    ) -> Mapping[BayesianBirdActions, Poss[BayesianBirdState]]:
        """ For each state, returns a dictionary U -> Possible Xs """
        # todo expand to allow other possibility monads

        if x.player_type == "0":
            possible = {}
            u = BayesianBirdActions(go=None)
            for _ in self.types:
                x2 = BayesianBirdState(z=x.z, stage=x.stage, player_type=_)
                possible[x2.player_type] = self.ps.unit(x2)
            return frozendict(possible)

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
    def successor(self, x: BayesianBirdState, u: BayesianBirdActions) -> BayesianBirdState:
        # trick to get unique NOT path dependent final states and
        # allow arbitrary payoff matrices
        altitude_incr: D = D(1) if x.stage == 0 else D(0.25)
        if u.go == UP:
            return replace(x, z=x.z + altitude_incr, stage=x.stage + 1)
        if u.go == DOWN:
            return replace(x, z=x.z - altitude_incr, stage=x.stage + 1)
        else:
            raise ZValueError(x=x, u=u)

    def get_shared_resources(self, x: X) -> FrozenSet[SR]:
        return None
        # raise NotImplementedError("For the toy example the concept of shared resources is not needed")

