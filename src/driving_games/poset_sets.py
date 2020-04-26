from dataclasses import dataclass
from typing import Callable, FrozenSet as ASet, Generic, Mapping, TypeVar

from frozendict import frozendict

from zuper_commons.types import check_isinstance, ZNotImplementedError
from zuper_typing import debug_print
from . import logger
from .poset import (COMP_OUTCOMES, ComparisonOutcome, FIRST_PREFERRED, INCOMPARABLE, INDIFFERENT, P, Preference,
                    SECOND_PREFERRED)

__all__ = ["SetPreference1"]


class SetPreference1(Preference[ASet[P]]):
    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self):
        return ASet[self.p0.get_type()]

    def __repr__(self):
        d = {"T": self.get_type(), "p0": self.p0}
        return "SetPreference1: " + debug_print(d)

    def compare(self, A: ASet[P], B: ASet[P]) -> ComparisonOutcome:
        check_isinstance(A, frozenset)
        check_isinstance(B, frozenset)
        if len(A) == 1 and len(B) == 1:
            a1 = list(A)[0]
            b1 = list(B)[0]
            res = self.p0.compare(a1, b1)
            assert res in COMP_OUTCOMES, (res, self.p0)
            return res

        return compare_sets(A, B, self.p0)

        if is_set_preferred_or_indifferent(A, B, self.p0):
            return FIRST_PREFERRED
        if is_set_preferred_or_indifferent(B, A, self.p0):
            return SECOND_PREFERRED

        logger.info('not implemented, using incomparable',
                    A=A, B=B)
        return INCOMPARABLE
        raise ZNotImplementedError(a=a, b=b)


def compare_sets(A: ASet[P], B: ASet[P], pref: Preference[P]) -> ComparisonOutcome:
    all_res = set()
    for a in A:
        res = set(pref.compare(a, b) for b in B)
        all_res.update(res)
        if INCOMPARABLE in res:
            return INCOMPARABLE

    if all_res == {INDIFFERENT}:
        return INDIFFERENT
    if all_res == {INDIFFERENT, FIRST_PREFERRED}:
        return FIRST_PREFERRED
    if all_res == {INDIFFERENT, SECOND_PREFERRED}:
        return SECOND_PREFERRED
    return INCOMPARABLE


#
#
# def is_set_preferred_or_indifferent(A: ASet[P], B: ASet[P], pref: Preference[P]) -> bool:
#     for a in A:
#         if not is_first_preferred_or_indifferent(a, B, pref):
#             return False
#     return True
#
# def is_first_preferred_or_indifferent(a: P, B: ASet[P], pref: Preference[P]) -> bool:
#     for b in B:
#         res = pref.compare(a, b)
#         if res not in [FIRST_PREFERRED, INDIFFERENT]:
#             return False
#     return True
#


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
