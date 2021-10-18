from decimal import Decimal as D
from typing import Type, Union

from zuper_commons.types import check_isinstance

from .preferences_base import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
)

__all__ = ["SmallerPreferred", "SmallerPreferredTol"]

Scalar = Union[D, float]


class SmallerPreferred(Preference[Scalar]):
    """
    The usual total order on the scalars.

    It is equivalent to `SmallerPreferredTol(0)`.

    """

    def get_type(self) -> Type[Scalar]:
        return Scalar

    def compare(self, a: Scalar, b: Scalar) -> ComparisonOutcome:
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


class SmallerPreferredTol(Preference[Scalar]):
    """ The semi-order on the scalar given a tolerance. """

    tol: Scalar

    def __init__(self, tol: Scalar):
        self.tol = tol

    def get_type(self) -> Type[Scalar]:
        return Scalar

    def compare(self, a: Scalar, b: Scalar) -> ComparisonOutcome:
        """ Returns :any:`INDIFFERENT` if `|a-b|<=tol`. """
        if abs(a - b) <= self.tol:
            return INDIFFERENT
        if a < b:
            return FIRST_PREFERRED
        if b < a:
            return SECOND_PREFERRED
        assert False, (a, b)

    def __repr__(self) -> str:
        return f"SmallerPreferredTol({self.tol})"
