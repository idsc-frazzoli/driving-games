from abc import abstractmethod
from functools import lru_cache
from typing import FrozenSet, Type

from possibilities.sets import SetPoss
from zuper_typing import debug_print
from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    P,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["SetDominancePreference", "SetWorstCasePreference", "compare_dominance_sets", "SetPreference"]


class SetPreference(Preference[SetPoss[P]]):
    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self) -> Type[SetPoss[P]]:
        return type(self)  # [self.p0.get_type()]

    def __repr__(self) -> str:
        d = {"T": self.get_type(), "p0": self.p0}
        return "SetPreference: " + debug_print(d)

    @abstractmethod
    def compare(self, A: SetPoss[P], B: SetPoss[P]) -> ComparisonOutcome:
        pass


class SetDominancePreference(SetPreference):
    def compare(self, A: SetPoss[P], B: SetPoss[P]) -> ComparisonOutcome:
        a_s = A.support()
        b_s = B.support()
        r = compare_dominance_sets(a_s, b_s, self.p0)
        # logger.info('SetPreference1', a_s=a_s, b_s=b_s, r=r)
        return r


@lru_cache(None)
def compare_dominance_sets(A: FrozenSet[P], B: FrozenSet[P], pref: Preference[P]) -> ComparisonOutcome:
    if A is B or (A == B):
        return INDIFFERENT

    has_first_preferred = False
    has_second_preferred = False
    all_res = set()
    for a in A:
        for b in B:
            r1 = pref.compare(a, b)
            all_res.add(r1)

            if r1 == INCOMPARABLE:
                return INCOMPARABLE
            if r1 == FIRST_PREFERRED:
                has_first_preferred = True
            if r1 == SECOND_PREFERRED:
                has_second_preferred = True
            if has_first_preferred and has_second_preferred:
                return INCOMPARABLE

    if all_res == {INDIFFERENT}:
        return INDIFFERENT
    if all_res == {INDIFFERENT, FIRST_PREFERRED} or all_res == {FIRST_PREFERRED}:
        return FIRST_PREFERRED
    if all_res == {INDIFFERENT, SECOND_PREFERRED} or all_res == {SECOND_PREFERRED}:
        return SECOND_PREFERRED
    assert False, all_res


class SetWorstCasePreference(SetPreference):
    def compare(self, A: SetPoss[P], B: SetPoss[P]) -> ComparisonOutcome:
        a_s = A.support()
        b_s = B.support()
        r = compare_worst_case_sets(a_s, b_s, self.p0)
        # logger.info('SetPreference1', a_s=a_s, b_s=b_s, r=r)
        return r


def _find_worst_case_set(A: FrozenSet[P], pref: Preference[P]) -> P:
    worst_a: P = None
    for a in A:
        if worst_a is None:
            worst_a = a
        else:
            r = pref.compare(worst_a, a)
            if r == FIRST_PREFERRED:
                worst_a = a
    return worst_a


@lru_cache(None)
def compare_worst_case_sets(A: FrozenSet[P], B: FrozenSet[P], pref: Preference[P]) -> ComparisonOutcome:
    if A is B or (A == B):
        return INDIFFERENT
    worst_a = _find_worst_case_set(A, pref)
    worst_b = _find_worst_case_set(B, pref)
    r = pref.compare(worst_a, worst_b)
    return r
