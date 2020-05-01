from functools import lru_cache
from typing import FrozenSet, Type

from possibilities import One, Poss
from zuper_typing import debug_print
from . import logger
from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    P,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["SetPreference1", "compare_sets"]


class SetPreference1(Preference[Poss[P, One]]):
    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self) -> Type[Poss[P, One]]:
        return Poss  # [self.p0.get_type()]

    def __repr__(self) -> str:
        d = {"T": self.get_type(), "p0": self.p0}
        return "SetPreference1: " + debug_print(d)

    # @lru_cache(None)
    def compare(self, A: Poss[P, One], B: Poss[P, One]) -> ComparisonOutcome:
        # check_poss(A)
        # check_poss(B)
        # if len(A) == 1 and len(B) == 1:
        #     a1 = list(A)[0]
        #     b1 = list(B)[0]
        #     res = self.p0.compare(a1, b1)
        #     assert res in COMP_OUTCOMES, (res, self.p0)
        #     return res
        a_s = A.support()
        b_s = B.support()
        r = compare_sets(a_s, b_s, self.p0)
        # logger.info('SetPreference1', a_s=a_s, b_s=b_s, r=r)
        return r

        # return compare_sets_cached(A.support(), B.support(), self.p0)


class CompareCache:
    cache = {}


def compare_sets_cached(A: FrozenSet[P], B: FrozenSet[P], pref: Preference[P]) -> ComparisonOutcome:
    sa = repr(A)
    sb = repr(B)
    key = sa, sb
    if key not in CompareCache.cache:
        res = compare_sets(A, B, pref)
        CompareCache.cache[key] = res
    return CompareCache.cache[key]


@lru_cache(None)
def compare_sets(A: FrozenSet[P], B: FrozenSet[P], pref: Preference[P]) -> ComparisonOutcome:
    if A is B:
        return INDIFFERENT

    has_first_preferred = False
    has_second_preferred = False
    all_res = set()
    for a in A:
        for b in B:
            r1 = pref.compare(a, b)
            all_res.add(r1)

            if r1 == INCOMPARABLE:
                # logger.info('compare_sets', a=a, b=b, r1=r1, all_res=all_res)
                return INCOMPARABLE
            if r1 == FIRST_PREFERRED:
                has_first_preferred = True
            if r1 == SECOND_PREFERRED:
                has_second_preferred = True
            if has_first_preferred and has_second_preferred:
                # logger.info('both', a=a, b=b, r1=r1, all_res=all_res)
                return INCOMPARABLE

    if all_res == {INDIFFERENT}:
        return INDIFFERENT
    if all_res == {INDIFFERENT, FIRST_PREFERRED} or all_res == {FIRST_PREFERRED}:
        return FIRST_PREFERRED
    if all_res == {INDIFFERENT, SECOND_PREFERRED} or all_res == {SECOND_PREFERRED}:
        return SECOND_PREFERRED
    assert False, all_res
