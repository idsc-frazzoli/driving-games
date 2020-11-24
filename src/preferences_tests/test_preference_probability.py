from fractions import Fraction

from driving_games import VehicleCosts
from possibilities import ProbDist, Poss
from preferences.preferences_probability import ProbPrefExpectedValue
from preferences_tests import logger


def test_expected_value():
    VehicleCostsDist = ProbDist[Poss[VehicleCosts]]
    mapping = {1: Fraction(1, 3), 2: Fraction(2, 3)}
    dist = VehicleCostsDist(mapping)
    dist_expectation = ProbPrefExpectedValue._expected(dist)
    logger.info(dist_expectation)
