import itertools
from typing import Callable, Collection, Iterator, Mapping, TypeVar

from frozendict import frozendict
from toolz import keyfilter, valfilter, valmap

K = TypeVar("K")
V = TypeVar("V")
W = TypeVar("W")


def iterate_dict_combinations(a: Mapping[K, Collection[V]]) -> Iterator[Mapping[K, V]]:
    ks = list(a)
    vs = [a[_] for _ in ks]
    alls = list(itertools.product(*tuple(vs)))
    for x in alls:
        d = frozendict(zip(ks, x))
        yield d


def fkeyfilter(pred: Callable[[K], bool], a: Mapping[K, V]) -> Mapping[K, V]:
    """ Wrapper around `toolz.keyfilter`. Adds frozendict, and helps with types."""
    return frozendict(keyfilter(pred, a))


def fvalfilter(pred: Callable[[V], bool], a: Mapping[K, V]) -> Mapping[K, V]:
    """ Wrapper around `toolz.valfilter`. Adds frozendict, and helps with types."""
    return frozendict(valfilter(pred, a))


def fvalmap(pred: Callable[[V], W], a: Mapping[K, V]) -> Mapping[K, W]:
    """ Wrapper around `toolz.keyfilter`. Adds frozendict, and helps with types."""
    return frozendict(valmap(pred, a))
