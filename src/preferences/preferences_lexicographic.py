from typing import Any, Tuple, Type

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


class LexicographicPreference(Preference[Tuple[Any, ...]]):
    prefs: Tuple[Preference[Any], ...]

    def __init__(self, prefs: Tuple[Preference[Any], ...]):
        self.prefs = prefs

    def __repr__(self) -> str:
        d = {"T": self.get_type(), "prefs": self.prefs}
        return "LexicographicPreference:\n" + debug_print(d)

    def get_type(self) -> Type[Tuple[Preference[Any], ...]]:
        t = make_Tuple(*tuple(_.get_type() for _ in self.prefs))
        return t

    def compare(self, a: Tuple[Any, ...], b: Tuple[Any, ...]) -> ComparisonOutcome:
        # check_isinstance(a, tuple)
        # check_isinstance(b, tuple)
        n = len(self.prefs)
        for i in range(n):
            c_i = self.prefs[i].compare(a[i], b[i])
            assert c_i in COMP_OUTCOMES, c_i
            if c_i in (FIRST_PREFERRED, SECOND_PREFERRED, INCOMPARABLE):
                return c_i

        return INDIFFERENT
