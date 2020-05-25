from typing import Mapping, Tuple, Type, TypeVar

from zuper_commons.types import ZValueError
from zuper_typing import debug_print
from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INCOMPARABLE,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["StrictProductPreference", "StrictProductPreferenceDict"]

V = TypeVar("V")


class StrictProductPreference(Preference[V]):
    """ Computes the product of the preferences. """

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
        """
        Returns the product of the preferences specified in the constructor.

        This returns:

        - :any:`FIRST_PREFERRED` if all voices agree on either :any:`FIRST_PREFERRED` or :any:`INDIFFERENT`.
        - :any:`SECOND_PREFERRED` if all voices agree on either :any:`SECOND_PREFERRED` or :any:`INDIFFERENT`.
        - :any:`INDIFFERENT` if it is :any:`INDIFFERENT` for everybody.
        - :any:`INCOMPARABLE` if there is any :any:`INCOMPARABLE` or both :any:`FIRST_PREFERRED`  and
          :any:`SECOND_PREFERRED`.


        """
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


K = TypeVar("K")


class StrictProductPreferenceDict(Preference[Mapping[K, V]]):
    """ Computes the product of the preferences. """

    prefs: Mapping[K, Preference[V]]

    def __init__(self, prefs: Mapping[K, Preference[V]]):
        self.prefs = prefs

    def get_type(self) -> Type[V]:
        T = list(self.prefs.values())[0].get_type()
        return T

    def __repr__(self) -> str:
        r = {"T": self.get_type(), "prefs": self.prefs}
        return "StrictProductPreference\n" + debug_print(r)

    def compare(self, a: Mapping[K, V], b: Mapping[K, V]) -> ComparisonOutcome:
        """
        Returns the product of the preferences specified in the constructor.

        This returns:

        - :any:`FIRST_PREFERRED` if all voices agree on either :any:`FIRST_PREFERRED` or :any:`INDIFFERENT`.
        - :any:`SECOND_PREFERRED` if all voices agree on either :any:`SECOND_PREFERRED` or :any:`INDIFFERENT`.
        - :any:`INDIFFERENT` if it is :any:`INDIFFERENT` for everybody.
        - :any:`INCOMPARABLE` if there is any :any:`INCOMPARABLE` or both :any:`FIRST_PREFERRED`  and
          :any:`SECOND_PREFERRED`.


        """

        keys_prefs = set(self.prefs)
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        if not (keys_prefs == keys_a == keys_b):
            msg = "Mismatch of keys"
            raise ZValueError(msg, _self=self, a=a, b=b)

        # check_isinstance(a, dict, _self=self)
        # check_isinstance(b, dict, _self=self)
        outcomes = []
        # logger.info('Product', a=a, b=b)
        for key in keys_prefs:
            pref = self.prefs[key]
            r = pref.compare(a[key], b[key])
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
