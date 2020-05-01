from typing import Callable, Dict, Mapping, TypeVar

import toolz

__all__ = []

A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")


def valmap(f: Callable[[A], B], d: Mapping[K, A]) -> Dict[K, B]:
    return toolz.valmap(f, d)
