from typing import Tuple, TypeVar
from driving_games import logger
from zuper_commons.types import check_isinstance
from zuper_typing import debug_print, make_Tuple
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


K = TypeVar("K")
V = TypeVar("V")


class StrictProductPreference(Preference[V]):
    prefs: Tuple[Preference[V], ...]

    def __init__(self, prefs: Tuple[Preference[V], ...]):
        self.prefs = prefs

    def get_type(self):
        T = self.prefs[0].get_type()
        return T

    def __repr__(self):
        r = {"T": self.get_type(), "prefs": self.prefs}
        return "StrictProductPreference\n" + debug_print(r)

    def compare(self, a: V, b: V) -> ComparisonOutcome:
        # check_isinstance(a, dict, _self=self)
        # check_isinstance(b, dict, _self=self)
        outcomes = []
        for pref in self.prefs:
            r = pref.compare(a, b)
            outcomes.append(r)

        # logger.info(outcomes=outcomes)
        # - any incomparable -> incomparable
        # - no incomparable:
        #       - all indifferent: INDIFFERENT
        #       - all first -> first
        #       - all second -> second
        #       - a mix of first, second, indifferent -> incomparable
        o = set(outcomes)
        if INCOMPARABLE in o:
            return INCOMPARABLE
        if o == {FIRST_PREFERRED} or o == {FIRST_PREFERRED, INDIFFERENT}:
            return FIRST_PREFERRED
        if o == {SECOND_PREFERRED} or o == {SECOND_PREFERRED, INDIFFERENT}:
            return SECOND_PREFERRED
        if o == {INDIFFERENT}:
            return INDIFFERENT
        return INCOMPARABLE
