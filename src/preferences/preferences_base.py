from abc import ABC, abstractmethod
from typing import (
    Generic,
    NewType,
    Type,
    TypeVar,
)

__all__ = [
    "COMP_OUTCOMES",
    "ComparisonOutcome",
    "INCOMPARABLE",
    "FIRST_PREFERRED",
    "SECOND_PREFERRED",
    "Preference",
    "INDIFFERENT",
]

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
