from fractions import Fraction

from possibilities import ProbDist
from possibilities.prob import expected_value
from preferences_tests import logger


def test_expected_value():
    dist = ProbDist({1: Fraction(1, 3), 2: Fraction(2, 3)})
    dist_expectation = expected_value(dist)
    logger.info(dist_expectation)
