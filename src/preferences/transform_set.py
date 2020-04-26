from dataclasses import dataclass
from typing import Callable, FrozenSet as ASet, Generic, TypeVar

__all__ = ["TransformSet"]

A = TypeVar("A")
B = TypeVar("B")


@dataclass
class TransformSet(Generic[A, B]):
    convert: Callable[[A], B]

    def __call__(self, X: ASet[A]) -> ASet[B]:
        res = set()
        for a in X:
            b = self.convert(a)
            res.add(b)
        return frozenset(res)
