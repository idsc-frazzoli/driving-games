import itertools
from typing import Collection, Iterator, Mapping, TypeVar

from frozendict import frozendict

K = TypeVar("K")
V = TypeVar("V")


def iterate_dict_combinations(a: Mapping[K, Collection[V]]) -> Iterator[Mapping[K, V]]:
    ks = list(a)
    vs = [a[_] for _ in ks]
    alls = list(itertools.product(*tuple(vs)))
    for x in alls:
        d = frozendict(zip(ks, x))
        yield d
