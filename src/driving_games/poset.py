from abc import ABC, abstractmethod
from decimal import Decimal as D
from typing import (
    Generic,
    NewType,
    Type,
    TypeVar,
)

P = TypeVar("P")

ComparisonOutcome = NewType("ComparisonOutcome", str)
INCOMPARABLE = ComparisonOutcome("incomparable")
INDIFFERENT = ComparisonOutcome("indifferent")
FIRST_PREFERRED = ComparisonOutcome("first_preferred")
SECOND_PREFERRED = ComparisonOutcome("second_preferred")
COMP_OUTCOMES = (INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, SECOND_PREFERRED)


class Preference(Generic[P], ABC):
    @abstractmethod
    def get_type(self) -> Type[P]:
        ...

    @abstractmethod
    def compare(self, a: P, b: P) -> ComparisonOutcome:
        ...


class SmallerPreferred(Preference[D]):
    def get_type(self) -> Type[P]:
        return D

    def compare(self, a: D, b: D) -> ComparisonOutcome:
        if a == b:
            return INDIFFERENT
        if a < b:
            return FIRST_PREFERRED
        if b < a:
            return SECOND_PREFERRED
        assert False, (a, b)

    def __repr__(self):
        return "SmallerPreferred"
