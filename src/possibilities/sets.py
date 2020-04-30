import itertools
from typing import Callable, cast, Collection, FrozenSet, Iterable, NewType, Tuple, TypeVar

from frozendict import frozendict
from numpy.random.mtrand import RandomState

from .base import PossibilityStructure, Sampler
from .poss import Poss
from .utils import non_empty_sets

__all__ = ["ProbabilitySet"]

A = TypeVar("A")
B = TypeVar("B")

One = NewType("One", str)

one = cast(One, ())


class ProbabilitySet(PossibilityStructure[One]):
    def lift_one(self, a: A) -> Poss[A, One]:
        return self.lift_many([a])

    def lift_many(self, a: Collection[A]) -> Poss[A, One]:
        elements = set(a)
        x = {_: one for _ in elements}
        return Poss(frozendict(x))

    def build(self, a: Poss[A, One], f: Callable[[A], B]) -> Poss[B, One]:
        res = set(f(_) for _ in a.support())
        return self.lift_many(res)

    def flatten(self, a: Poss[Poss[A, One], One]) -> Poss[A, One]:
        supports = [_.support() for _ in a.support()]
        s = set(itertools.chain.from_iterable(supports))
        return self.lift_many(s)

    def get_sampler(self, seed: int) -> "SetSampler[One]":
        return SetSampler(seed)

    def mix(self, a: Collection[A]) -> FrozenSet[Poss[A, One]]:
        poss = non_empty_sets(frozenset(a))
        return frozenset(map(self.lift_many, poss))

    def multiply(self, a: Iterable[One]) -> One:
        a = set(a)
        assert a == {one}, (a, one)
        return one

    def fold(self, a: Iterable[Tuple[A, One]]) -> Poss[A, One]:
        res = set(x for x, _ in a)
        return self.lift_many(res)


class SetSampler(Sampler[One]):
    def __init__(self, seed: int):
        self.rs = RandomState(seed)

    def sample(self, options: Poss[A, One]) -> A:
        support = list(options.support())
        indices = list(range(len(support)))
        i = self.rs.choice(indices, 1, replace=False)
        return support[int(i)]
