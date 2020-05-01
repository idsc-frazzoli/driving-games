from typing import Mapping, TypeVar

from frozendict import frozendict

from .preferences_base import FIRST_PREFERRED, Preference, SECOND_PREFERRED

__all__ = ["remove_dominated"]

A = TypeVar("A")
B = TypeVar("B")


def remove_dominated(orig: Mapping[A, B], pref: Preference[B]) -> Mapping[A, B]:
    # logger.info(orig=orig, pref=pref)

    def is_strictly_dominated(x: B) -> bool:
        for k1, v1 in orig.items():
            r = pref.compare(v1, x)
            if r == FIRST_PREFERRED:
                return True
        return False

    res = {}
    for k, v in orig.items():
        if not is_strictly_dominated(v):
            res[k] = v
    return frozendict(res)


def worst_cases(orig: Mapping[A, B], pref: Preference[B]) -> Mapping[A, B]:
    # logger.info(orig=orig, pref=pref)

    def something_is_worst(x: B) -> bool:
        for k1, v1 in orig.items():
            r = pref.compare(v1, x)
            if r == SECOND_PREFERRED:
                return True
        return False

    res = {}
    for k, v in orig.items():
        if not something_is_worst(v):
            res[k] = v
    return frozendict(res)
