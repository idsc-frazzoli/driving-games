from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Generic, Type, TypeVar

from . import logger
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
        ret = self.p0.compare(x1, y1)
        # logger.info('PrefConverter', x=x, y=y, y1=y1, x1=x1, ret=ret)
        return ret
