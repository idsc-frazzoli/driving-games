from typing import FrozenSet as ASet, Type

from zuper_commons.types import check_isinstance
from zuper_typing import debug_print

from .preferences_base import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    P,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["SetPreference1", "compare_sets"]


class SetPreference1(Preference[ASet[P]]):
    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self) -> Type[ASet[P]]:
        return ASet[self.p0.get_type()]

    def __repr__(self) -> str:
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
