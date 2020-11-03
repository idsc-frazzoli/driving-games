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
""" The type of comparison outcomes """

INCOMPARABLE = ComparisonOutcome("incomparable")
""" The two options are incomparable. """

INDIFFERENT = ComparisonOutcome("indifferent")
""" We are indifferent among the two options. """

FIRST_PREFERRED = ComparisonOutcome("first_preferred")
""" We prefer the first option. """

SECOND_PREFERRED = ComparisonOutcome("second_preferred")
""" We prefer the second option. """

COMP_OUTCOMES = (INDIFFERENT, INCOMPARABLE, FIRST_PREFERRED, SECOND_PREFERRED)
""" All possible outcomes of a comparison.  """


class Preference(Generic[P], ABC):
    """
    The abstract interfaces for a preference relation.
    A preference relation is reflexive. No other property is assumed.
    """

    @abstractmethod
    def get_type(self) -> Type[P]:
        ...

    @abstractmethod
    def compare(self, a: P, b: P) -> ComparisonOutcome:
        """

        Compares two possibilities.

        It returns one of 4 outcomes:

        - :any:`INDIFFERENT`: We are indifferent among the two options.
        - :any:`INCOMPARABLE`: The two options are incomparable.
        - :any:`FIRST_PREFERRED`: We prefer the first option.
        - :any:`SECOND_PREFERRED`:  We prefer the second option.

        """
