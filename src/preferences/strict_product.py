from typing import Tuple, Type, TypeVar

from zuper_typing import debug_print
from . import logger
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
        # logger.info('Product', a=a, b=b)
        for i, pref in enumerate(self.prefs):
            r = pref.compare(a, b)
            # logger.info(pref=pref, i=i,  r=r)
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
            r = INCOMPARABLE
        elif o == {FIRST_PREFERRED} or o == {FIRST_PREFERRED, INDIFFERENT}:
            r = FIRST_PREFERRED
        elif o == {SECOND_PREFERRED} or o == {SECOND_PREFERRED, INDIFFERENT}:
            r = SECOND_PREFERRED
        elif o == {INDIFFERENT}:
            r = INDIFFERENT
        else:
            r = INCOMPARABLE

        # logger.info('StrictProduct', a=a, b=b, outcomes=outcomes, r=r, prefs=self.prefs)
        return r
