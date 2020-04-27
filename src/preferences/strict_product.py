from typing import Tuple, Type, TypeVar

from zuper_typing import debug_print

from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["StrictProductPreference"]

V = TypeVar("V")


class StrictProductPreference(Preference[V]):
    prefs: Tuple[Preference[V], ...]

    def __init__(self, prefs: Tuple[Preference[V], ...]):
        self.prefs = prefs

    def get_type(self) -> Type[V]:
        T = self.prefs[0].get_type()
        return T

    def __repr__(self) -> str:
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
