from abc import abstractmethod
from typing import Type
from decimal import Decimal as D
from zuper_typing import debug_print

from possibilities.prob import ProbPoss
from preferences import Preference, ComparisonOutcome
from preferences.preferences_base import P


class ProbPreference(Preference[ProbPoss[P]]):
    """"""

    p0: Preference[P]

    def __init__(self, p0: Preference[P]):
        self.p0 = p0

    def get_type(self) -> Type[ProbPoss[P]]:
        return ProbPoss

    def __repr__(self) -> str:
        d = {"T": self.get_type(), "p0": self.p0}
        return "ProbPreference: " + debug_print(d)

    @abstractmethod
    def compare(self, A: ProbPoss[P], B: ProbPoss[P]) -> ComparisonOutcome:
        pass


class ProbPrefExpectedValue(ProbPreference):
    # todo can be more elegant and return Xpercentile with x=.5
    def compare(self, A: ProbPoss[P], B: ProbPoss[P]) -> ComparisonOutcome:
        expected_A = self._expected(A)
        expected_B = self._expected(B)
        return self.p0.compare(expected_A, expected_B)

    @staticmethod
    def _expected(A: ProbPoss[P]) -> P:
        expected_A = None
        for a, prob in A.it():
            if expected_A is None:
                expected_A = a * prob
            else:
                expected_A += a * prob
        return expected_A


class ProbPrefXPercentile(ProbPreference):
    def __init__(self, x: D):
        self.x: D = x
        raise NotImplementedError

    def compare(self, A: ProbPoss[P], B: ProbPoss[P]) -> ComparisonOutcome:
        x_percentile_A = ...
        x_percentile_B = ...
        return self.p0.compare(x_percentile_A, x_percentile_B)


class ProbPrefWorstCase(ProbPreference):
    def compare(self, A: ProbPoss[P], B: ProbPoss[P]) -> ComparisonOutcome:
        raise NotImplementedError
        worst_A = ...
        worst_B = ...
        return self.p0.compare(worst_A, worst_B)


class ProbPrefStochasticDominance(ProbPreference):
    def compare(self, A: ProbPoss[P], B: ProbPoss[P]) -> ComparisonOutcome:
        # todo
        raise NotImplementedError
