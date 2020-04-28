from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

from zuper_commons.types import check_isinstance

PD = TypeVar('PD')
PV = TypeVar('PV')


@dataclass
class Poss(Generic[PD, PV]):
    p: Mapping[PD, PV]

    def __post_init__(self):
        check_isinstance(self.p, frozendict)
    def check_contains(self, T: type):
        for _ in self.p:
            check_isinstance(_, T, _=self)
