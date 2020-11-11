import itertools
from dataclasses import dataclass
from typing import Callable, Collection, FrozenSet, Mapping, Set, TypeVar

from frozendict import frozendict
from numpy.random.mtrand import RandomState

from zuper_commons.types import ZValueError
from .base import PossibilityMonad, Sampler
from .poss import Poss
from .utils import non_empty_sets

__all__ = ["PossibilitySet"]

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


@dataclass(unsafe_hash=True)
class SetPoss(Poss[A]):
    _p: FrozenSet[A]

    def __post_init__(self) -> None:
        self._r = None

    def __repr__(self) -> str:
        if self._r is None:
            self._r = f"Set({self._p.__repr__()})"
        return self._r

    def support(self) -> FrozenSet[A]:
        """ Returns the support of the distribution """
        return self._p


def make_setposs(f: FrozenSet[A]) -> SetPoss[A]:
    return SetPoss(f)
    # if f not in Cache.cache:
    #     # logger.info(f=f)
    #     Cache.cache[f] = SetPoss(f)
    # return Cache.cache[f]


class PossibilitySet(PossibilityMonad):
    def unit(self, a: A) -> SetPoss[A]:
        return self.lift_many([a])

    def lift_many(self, a: Collection[A]) -> SetPoss[A]:
        elements = frozenset(a)
        return make_setposs(elements)

    def build(self, a: SetPoss[A], f: Callable[[A], B]) -> SetPoss[B]:
        res = list(f(_) for _ in a.support())
        try:
            res = set(res)
        except TypeError as e:
            msg = "Function gave a result that is not hashable"
            raise ZValueError(msg, f=f, res=res) from e
        return self.lift_many(res)

    def build_multiple(self, a: Mapping[K, SetPoss[A]], f: Callable[[Mapping[K, A]], B]) -> SetPoss[B]:
        sources = list(a)
        supports = [a[_].support() for _ in sources]
        res: Set[B] = set()
        for _ in itertools.product(*tuple(supports)):
            elements = frozendict(zip(sources, _))
            r = f(elements)
            res.add(r)

        return make_setposs(frozenset(res))

    def join(self, a: SetPoss[SetPoss[A]]) -> SetPoss[A]:
        supports = [_.support() for _ in a.support()]
        s = set(itertools.chain.from_iterable(supports))
        return self.lift_many(s)

    def get_sampler(self, seed: int) -> "SetSampler":
        return SetSampler(seed)

    def mix(self, a: Collection[A]) -> FrozenSet[SetPoss[A]]:
        poss = non_empty_sets(frozenset(a))
        return frozenset(map(self.lift_many, poss))


class SetSampler(Sampler):
    def __init__(self, seed: int):
        self.rs = RandomState(seed)

    def sample(self, options: SetPoss[A]) -> A:
        support = list(options.support())
        indices = list(range(len(support)))
        i = self.rs.choice(indices, 1, replace=False)
        return support[int(i)]
