from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Type, TypeVar

from .preferences_base import ComparisonOutcome, Preference

__all__ = ["PrefConverter"]
A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class PrefConverter(Preference[A], Generic[A, B]):
    A: Type[A]
    B: Type[B]
    convert: Callable[[A], B]
    p0: Preference[B]

    def get_type(self) -> Type[B]:
        return self.B

    @lru_cache(None)
    def compare(self, x: A, y: A) -> ComparisonOutcome:
        x1: B = self.convert(x)
        y1: B = self.convert(y)
        return self.p0.compare(x1, y1)
