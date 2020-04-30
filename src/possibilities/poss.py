from dataclasses import dataclass
from typing import FrozenSet, Generic, Iterator, Mapping, Tuple, Type, TypeVar

from frozendict import frozendict
from zuper_commons.types import check_isinstance

__all__ = ["Poss", "Φ", "check_poss"]

Φ = TypeVar("Φ")
A = TypeVar("A")


@dataclass(frozen=True)
class Poss(Generic[A, Φ]):
    p: Mapping[A, Φ]

    def __post_init__(self):
        check_isinstance(self.p, frozendict)

    def check_contains(self, T: type, **kwargs):
        for _ in self.p:
            check_isinstance(_, T, poss=self, **kwargs)

    def it(self) -> Iterator[Tuple[A, Φ]]:
        for _ in self.p.items():
            yield _

    def support(self) -> FrozenSet[A]:
        """ Returns the support of the distribution """
        return frozenset(self.p)

    def get(self, a: A) -> Φ:
        return self.p[a]


def check_poss(a: Poss[A, Φ], T: Type[A] = object, **kwargs):
    check_isinstance(a, Poss, **kwargs)
    a.check_contains(T, **kwargs)
