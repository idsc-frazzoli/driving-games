import itertools
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce, lru_cache
from itertools import permutations
from math import isclose
from operator import add
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
from zuper_commons.types import ZTypeError

from games import GameConstants
from .base import PossibilityMonad, Sampler
from .poss import Poss

__all__ = ["ProbDist", "PossibilityDist", "ProbSampler", "expected_value"]

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


@dataclass(unsafe_hash=True)
class ProbDist(Poss[A]):
    p: Mapping[A, Fraction]
    __print_order__ = ["p"]
    _support: FrozenSet[A] = None
    _range: FrozenSet[Fraction] = None

    def __post_init__(self) -> None:
        self._support = frozenset(self.p)
        self._range = frozenset(self.p.values())
        # check_isinstance(self.p, frozendict)

    def it(self) -> Iterator[Tuple[A, Fraction]]:
        for _ in self.p.items():
            yield _

    def support(self) -> FrozenSet[A]:
        """Returns the support of the distribution"""
        return self._support

    def get(self, a: A) -> Fraction:
        return self.p[a]

    def __eq__(self, other: "ProbDist") -> bool:
        if self._support != other._support:
            return False
        if self._range != other._range:
            return False
        return self.p == other.p


def expected_value(dist: ProbDist[A]) -> A:
    """
    maybe this in the future will move to an enriched Dist that expands the current one with a bunch of operations.
    also it could be moved to be an independent method
    :return:
    """
    try:
        weighted = [a * w for a, w in dist.it()]
        return reduce(add, weighted)
    except TypeError as e:
        msg = (
            "\nThe current distribution does not seem to support the expected value operation."
            f"\nYou are trying to:\n{e.args}"
            f"\nCurrent distribution {dist.p}"
        )
        raise ZTypeError(msg=msg)


class PossibilityDist(PossibilityMonad):
    """Extension of the [distribution monad](https://ncatlab.org/nlab/show/distribution+monad#finite_distributions)."""

    def unit(self, a: A) -> ProbDist[A]:
        return self.lift_many([a])

    def lift_many(self, a: Collection[A]) -> ProbDist[A]:
        elements = list(a)
        n = len(elements)
        w = Fraction(1, n)
        x = {_: w for _ in elements}
        return ProbDist(frozendict(x))

    def join(self, a: ProbDist[ProbDist[A]]) -> ProbDist[A]:
        res = defaultdict(Fraction)
        for dist, weight in a.it():
            for a, wa in dist.it():
                res[a] += weight * wa
        return ProbDist(frozendict(res))

    def build(self, a: ProbDist[A], f: Callable[[A], B]) -> ProbDist[B]:
        res = defaultdict(Fraction)
        for x, weight in a.it():
            y = f(x)
            res[y] += weight
        return ProbDist(frozendict(res))

    def build_multiple(self, a: Mapping[K, ProbDist[A]], f: Callable[[Mapping[K, A]], B]) -> ProbDist[B]:
        sources = list(a)
        supports = [a[_].support() for _ in sources]
        res: Dict[Mapping[K, A], Fraction] = defaultdict(Fraction)
        for _ in itertools.product(*tuple(supports)):
            elements = frozendict(zip(sources, _))
            probs = [a[source].get(elements[source]) for source in sources]
            weight = reduce(Fraction.__mul__, probs)
            r = f(elements)
            # for v in r:
            #     res[v] += weight
            res[r] += weight
        return ProbDist(frozendict(res))

    def get_sampler(self, seed: int) -> "ProbSampler":
        return ProbSampler(seed)

    def mix(self, a: Collection[A]) -> FrozenSet[ProbDist[A]]:
        l = list(a)
        n = len(l)
        res = set()
        for c in enumerate_prob_assignments(n):
            p = frozendict(valfilter(lambda _: _ > 0, dict(zip(l, c))))
            res.add(ProbDist(p))
        return frozenset(res)


@lru_cache(maxsize=128)
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


def check_prob_dist(prob_poss: ProbDist, **kwargs):
    """Checks consistency of a JointMixedActions variable."""
    if not GameConstants.checks:
        return
    cumulative_dist = sum(prob_poss.p.values())
    assert isclose(cumulative_dist, 1)  # probably also exact equality


class ProbSampler(Sampler):
    rs: RandomState

    def __init__(self, seed: int):
        self.rs = RandomState(seed)

    def sample(self, options: ProbDist[A]) -> A:
        support = []
        prob = []
        for a, b in options.it():
            support.append(a)
            prob.append(float(b))

        indices = list(range(len(support)))
        i = self.rs.choice(indices, 1, replace=False, p=prob)
        return support[int(i)]
