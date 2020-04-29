from dataclasses import dataclass
from typing import Callable, FrozenSet, Generic, Iterator, Mapping, Tuple, Type, TypeVar

from frozendict import frozendict
from zuper_commons.types import check_isinstance

PD = TypeVar('PD')
PV = TypeVar('PV')

Y = TypeVar('Y')


@dataclass
class Poss(Generic[PD, PV]):
    p: Mapping[PD, PV]

    def __post_init__(self):
        check_isinstance(self.p, frozendict)

    def check_contains(self, T: type):
        for _ in self.p:
            check_isinstance(_, T, _=self)

    def it(self) -> Iterator[Tuple[PD, PV]]:
        for _ in self.p.items():
            yield _

    def support(self) -> FrozenSet[PD]:
        """ Returns the support of the distribution """
        return frozenset(self.p)

    def build(self, f: Callable[[PD], Y]) -> "Poss[PD, Y]":
        raise NotImplementedError()


def check_poss(a: Poss[PD, PV], T: Type[PV]):
    check_isinstance(a, Poss)
    a.check_contains(T)
