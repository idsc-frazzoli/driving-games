from typing import Dict, Mapping, Tuple, TypeVar

from zuper_commons.types import check_isinstance
from zuper_typing import debug_print
from .poset import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

A = TypeVar("A")
B = TypeVar("B")
__all__ = ["LexicographicPreference"]


class LexicographicPreference(Preference[Tuple[A, B]]):
    prefs: Tuple[Preference[A], Preference[B]]

    def __init__(self, prefs: Tuple[Preference[A], Preference[B]]):
        self.prefs = prefs

    def __repr__(self):
        d = {'prefs': self.prefs}
        return 'LexicographicPreference:\n' + debug_print(d)

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

K = TypeVar('K')
V = TypeVar('V')

class StrictProductPreference(Preference[Mapping[K, V]]):
    prefs:  Mapping[K, Preference[V]]

    def __init__(self, prefs: Mapping[K, Preference[V]]):
        self.prefs = prefs

    def compare(self, a: Mapping[K, V], b: Mapping[K, V]) -> ComparisonOutcome:
        check_isinstance(a, dict)
        check_isinstance(b, dict)
        outcomes = {}
        for k, pref in self.prefs.items():
            r =  pref.compare(a[k], b[k])
            outcomes[k] = r

        # - any incomparable -> incomparable
        # - no incomparable:
        #       - all indifferent: INDIFFERENT
        #       - all first -> first
        #       - all second -> second
        #       - a mix of first, second, indifferent -> incomparable
        o = set(outcomes)
        if INCOMPARABLE in o:
            return INCOMPARABLE
        if o == {FIRST_PREFERRED}:
            return FIRST_PREFERRED
        if o == {SECOND_PREFERRED}:
            return SECOND_PREFERRED
        if o == {INDIFFERENT}:
            return INDIFFERENT
        return INCOMPARABLE
