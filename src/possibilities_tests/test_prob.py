from fractions import Fraction
from typing import Dict

from zuper_commons.types import ZValueError

from games import JointPureActions, PlayerName
from possibilities import Poss
from possibilities.prob import enumerate_prob_assignments, ProbabilityFraction, ProbPoss, A
from . import logger
from .test_sets import check_possibilities


def test_fraction():
    pmonad = ProbabilityFraction()
    check_possibilities(pmonad)


def test_prob_mix():
    a = "a"
    b = "b"
    S = [a, b]
    pmonad = ProbabilityFraction()
    r = pmonad.mix(S)
    logger.info(r=r)
    # assert_equal(r, {pmonad.lift_many({0}), pmonad.lift_many({1}), pmonad.lift_many({0, 1})})


def test_prob_mix4():
    S = set("abcd")
    pmonad = ProbabilityFraction()
    r = pmonad.mix(S)
    if len(r) != 23:
        raise ZValueError(S=S, r=r, l=len(r))


def test_prob1():
    for i in [1, 2, 3, 4, 5]:
        res = enumerate_prob_assignments(i)
        logger.info(i=i, res=res)


def test_build_multiple1():
    distribution = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    P1 = PlayerName("1")
    a: Dict[PlayerName, ProbPoss[A]]
    a = {P1: distribution}

    def f(x):
        x_squared = {key: pow(value, 2.0) for key, value in x.items()}
        return list(x_squared.values())

    b = ProbabilityFraction()
    dist = b.build_multiple(a, f)
    print(dist)


def test_build_multiple2():
    distribution1 = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    distribution2 = ProbPoss({1: Fraction(1, 3), 2: Fraction(2, 3)})

    a: Dict[PlayerName, ProbPoss[A]]
    a = {"1": distribution1, "2": distribution2}

    def f(x):
        x_squared = {key: pow(value, 2.0) for key, value in x.items()}
        return list(x_squared.values())

    b = ProbabilityFraction()
    dist = b.build_multiple(a, f)
    print(dist)
