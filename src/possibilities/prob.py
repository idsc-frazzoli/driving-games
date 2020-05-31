import itertools
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from itertools import permutations
from typing import (
    AbstractSet,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterator,
    Mapping,
    Tuple,
    TypeVar,
)

from frozendict import frozendict
from numpy.random.mtrand import RandomState
from toolz import valfilter

from contracts import check_isinstance
from .base import PossibilityStructure, Sampler
from .poss import Poss

__all__ = ["ProbabilityFraction"]

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


@dataclass(unsafe_hash=True)
class ProbPoss(Poss[A]):
    p: Mapping[A, Fraction]
    __print_order__ = ["p"]
    _support: FrozenSet[A] = None
    _range: FrozenSet[Fraction] = None

    def __post_init__(self) -> None:
        self._support = frozenset(self.p)
        self._range = frozenset(self.p.values())
        # check_isinstance(self.p, frozendict)

    def check_contains(self, T: type, **kwargs):
        for _ in self.p:
            check_isinstance(_, T, poss=self, **kwargs)

    def it(self) -> Iterator[Tuple[A, Fraction]]:
        for _ in self.p.items():
            yield _

    def support(self) -> FrozenSet[A]:
        """ Returns the support of the distribution """
        return self._support

    def get(self, a: A) -> Fraction:
        return self.p[a]

    def __eq__(self, other) -> bool:
        if self._support != other._support:
            return False
        if self._range != other._range:
            return False
        # return True ##### XXXXX
        return self.p == other.p


class ProbabilityFraction(PossibilityStructure):
    def lift_one(self, a: A) -> ProbPoss[A]:
        return self.lift_many([a])

    def lift_many(self, a: Collection[A]) -> ProbPoss[A]:
        elements = list(a)
        n = len(elements)
        w = Fraction(1, n)
        x = {_: w for _ in elements}
        return ProbPoss(frozendict(x))

    def flatten(self, a: ProbPoss[ProbPoss[A]]) -> ProbPoss[A]:
        res = defaultdict(Fraction)
        for dist, weight in a.it():
            for a, wa in dist.it():
                res[a] += weight * wa
        return ProbPoss(frozendict(res))

    def build(self, a: ProbPoss[A], f: Callable[[A], B]) -> ProbPoss[B]:
        res = defaultdict(Fraction)
        for x, weight in a.it():
            y = f(x)
            res[y] += weight
        return ProbPoss(frozendict(res))

    def build_multiple(self, a: Mapping[K, ProbPoss[A]], f: Callable[[Mapping[K, A]], B]) -> ProbPoss[B]:
        sources = list(a)
        supports = [_.support() for _ in sources]
        res: Dict[Mapping[K, A], Fraction] = defaultdict(Fraction)
        for _ in itertools.product(*tuple(supports)):
            elements = frozendict(zip(sources, _))
            probs = [a[source].get(elements[source]) for source in sources]
            weight = reduce(Fraction.__mul__, probs)
            r = f(elements)
            res[r] += weight

        return ProbPoss(frozendict(res))

    def get_sampler(self, seed: int) -> "ProbSampler":
        return ProbSampler(seed)

    def mix(self, a: Collection[A]) -> FrozenSet[ProbPoss[A]]:
        l = list(a)
        n = len(l)
        res = set()
        for c in enumerate_prob_assignments(n):
            p = frozendict(valfilter(lambda _: _ > 0, dict(zip(l, c))))
            res.add(ProbPoss(p))
        return frozenset(res)


def enumerate_prob_assignments(n: int) -> AbstractSet[Tuple[Fraction, ...]]:
    zero = Fraction(0)
    one = Fraction(1)
    half = Fraction(1, 2)
    third = Fraction(1, 3)
    fourth = Fraction(1, 4)
    if n == 1:
        cases = {(one,)}
    elif n == 2:
        cases = {(one, zero), (half, half)}
    elif n == 3:
        cases = [(one, zero, zero), (third, third, third), (third, 2 * third, zero)]
    elif n == 4:
        cases = [
            (one, zero, zero, zero),
            (half, fourth, fourth, zero),
            (half, half, zero, zero),
            (fourth, fourth, fourth, fourth),
        ]
    elif n == 5:
        f = Fraction(1, 5)
        cases = [
            (f, f, f, f, f),
            (2 * f, f, f, f, zero),
            (2 * f, 2 * f, f, zero, zero),
            (3 * f, f, f, zero, zero),
            (3 * f, 2 * f, zero, zero, zero),
            (4 * f, f, zero, zero, zero),
            (one, zero, zero, zero, zero),
        ]
    else:
        raise NotImplementedError(n)
    res = set()
    for c in cases:
        for _ in permutations(c, n):
            # a = permute(c, _)
            res.add(_)
    return res


class ProbSampler(Sampler):
    rs: RandomState

    def __init__(self, seed: int):
        self.rs = RandomState(seed)

    def sample(self, options: ProbPoss[A]) -> A:
        support = []
        prob = []
        for a, b in options.it():
            support.append(a)
            prob.append(float(b))

        indices = list(range(len(support)))
        i = self.rs.choice(indices, 1, replace=False, p=prob)
        return support[int(i)]
