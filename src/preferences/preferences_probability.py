from abc import abstractmethod
from typing import Type
from decimal import Decimal as D
from zuper_typing import debug_print

from possibilities.prob import ProbDist, expected_value
from preferences import Preference, ComparisonOutcome
from preferences.preferences_base import P


class ProbPreference(Preference[ProbDist[P]]):
    """How to compare prob distributions"""

    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self) -> Type[ProbDist[P]]:
        return ProbDist

    def __repr__(self) -> str:
        d = {"T": self.get_type(), "p0": self.p0}
        return "ProbPreference: " + debug_print(d)

    @abstractmethod
    def compare(self, A: ProbDist[P], B: ProbDist[P]) -> ComparisonOutcome:
        pass


class ProbPrefExpectedValue(ProbPreference):
    def compare(self, A: ProbDist[P], B: ProbDist[P]) -> ComparisonOutcome:
        expected_A = expected_value(A)
        expected_B = expected_value(B)
        return self.p0.compare(expected_A, expected_B)


class ProbPrefXPercentile(ProbPreference):
    def __init__(self, x: D):
        self.x: D = x
        raise NotImplementedError

    def compare(self, A: ProbDist[P], B: ProbDist[P]) -> ComparisonOutcome:
        x_percentile_A = ...
        x_percentile_B = ...
        return self.p0.compare(x_percentile_A, x_percentile_B)


class ProbPrefWorstCase(ProbPreference):
    def compare(self, A: ProbDist[P], B: ProbDist[P]) -> ComparisonOutcome:
        raise NotImplementedError
        worst_A = ...
        worst_B = ...
        return self.p0.compare(worst_A, worst_B)


class ProbPrefStochasticDominance(ProbPreference):
    def compare(self, A: ProbDist[P], B: ProbDist[P]) -> ComparisonOutcome:
        # todo
        raise NotImplementedError
