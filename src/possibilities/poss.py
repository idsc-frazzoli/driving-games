from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Generic, Iterator, Mapping, Tuple, Type, TypeVar

from zuper_commons.types import check_isinstance

__all__ = ["Poss", "Φ", "check_poss"]

Φ = TypeVar("Φ")
A = TypeVar("A")


class Poss(Generic[A, Φ], ABC):
    @abstractmethod
    def check_contains(self, T: type, **kwargs):
        pass

    @abstractmethod
    def support(self) -> FrozenSet[A]:
        pass


CHECK = False


def check_poss(a: Poss[A, Φ], T: Type[A] = object, **kwargs):
    if not CHECK:
        return
    check_isinstance(a, Poss, **kwargs)
    a.check_contains(T, **kwargs)
