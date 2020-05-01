from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Type, TypeVar

from .preferences_base import ComparisonOutcome, Preference

__all__ = ["PrefConverter"]
A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class PrefConverter(Preference[A], Generic[A, B]):
    AT: Type[A]
    BT: Type[B]
    convert: Callable[[A], B]
    p0: Preference[B]

    def get_type(self) -> Type[B]:
        return self.BT

    @lru_cache(None)
    def compare(self, x: A, y: A) -> ComparisonOutcome:
        c: Callable[[A], B] = self.convert
        x1: B = c(x)
        y1: B = c(y)
        return self.p0.compare(x1, y1)
