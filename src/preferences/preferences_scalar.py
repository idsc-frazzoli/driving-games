from decimal import Decimal as D
from typing import Type

from zuper_commons.types import check_isinstance
from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["SmallerPreferred", "SmallerPreferredTol"]


class SmallerPreferred(Preference[D]):
    """
        The usual total order on the scalars.

        It is equivalent to `SmallerPreferredTol(0)`.

    """

    def get_type(self) -> Type[D]:
        return D

    def compare(self, a: D, b: D) -> ComparisonOutcome:
        """ """
        if a == b:
            return INDIFFERENT
        elif a < b:
            return FIRST_PREFERRED
        elif b < a:
            return SECOND_PREFERRED
        else:
            assert False, (a, b)

    def __repr__(self) -> str:
        return "SmallerPreferred"


class SmallerPreferredTol(Preference[D]):
    """ The semi-order on the scalar given a tolerance. """

    tol: D

    def __init__(self, tol: D):
        self.tol = tol

    def get_type(self) -> Type[D]:
        return D

    def compare(self, a: D, b: D) -> ComparisonOutcome:
        """ Returns :any:`INDIFFERENT` if `|a-b|<=tol`. """
        check_isinstance(a, D)
        check_isinstance(b, D)
        if abs(a - b) <= self.tol:
            return INDIFFERENT
        if a < b:
            return FIRST_PREFERRED
        if b < a:
            return SECOND_PREFERRED
        assert False, (a, b)

    def __repr__(self) -> str:
        return f"SmallerPreferredTol({self.tol})"
