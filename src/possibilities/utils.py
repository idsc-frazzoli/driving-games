from itertools import chain, combinations
from typing import FrozenSet, TypeVar

__all__ = ["non_empty_sets"]

A = TypeVar("A")


def non_empty_sets(iterable: FrozenSet[A]) -> FrozenSet[FrozenSet[A]]:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return frozenset(map(frozenset, chain.from_iterable(combinations(xs, n) for n in range(1, len(xs) + 1))))
