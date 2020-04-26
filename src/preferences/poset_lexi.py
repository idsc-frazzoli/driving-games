from typing import Tuple, TypeVar

from zuper_commons.types import check_isinstance
from zuper_typing import debug_print, make_Tuple

from .preferences_base import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["LexicographicPreference"]


A = TypeVar("A")
B = TypeVar("B")


class LexicographicPreference(Preference[Tuple[A, B]]):
    prefs: Tuple[Preference[A], Preference[B]]

    def __init__(self, prefs: Tuple[Preference[A], Preference[B]]):
        self.prefs = prefs

    def __repr__(self):
        d = {"T": self.get_type(), "prefs": self.prefs}
        return "LexicographicPreference:\n" + debug_print(d)

    def get_type(self):
        t = make_Tuple(*tuple(_.get_type() for _ in self.prefs))
        return t

    def compare(self, a: Tuple[A, B], b: Tuple[A, B]) -> ComparisonOutcome:
        check_isinstance(a, tuple)
        check_isinstance(b, tuple)
        n = len(self.prefs)
        for i in range(n):
            c_i = self.prefs[i].compare(a[i], b[i])
            assert c_i in COMP_OUTCOMES, c_i
            if c_i in (FIRST_PREFERRED, SECOND_PREFERRED, INCOMPARABLE):
                return c_i

        return INDIFFERENT
