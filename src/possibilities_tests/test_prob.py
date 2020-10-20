from fractions import Fraction
from typing import Dict

from nose.tools import eq_
from zuper_commons.types import ZValueError

from games import PlayerName
from games.utils import valmap
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
    dist = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    result = ProbPoss({frozenset({1}): Fraction(1, 2), frozenset({4}): Fraction(1, 2)})
    P1 = PlayerName("1")
    a: Dict[PlayerName, ProbPoss[A]]
    a = {P1: dist}

    def f(x):
        return frozenset(valmap(lambda x: x ** 2, x).values())

    ps = ProbabilityFraction()
    dist = ps.build_multiple(a, f)
    eq_(dist, result)


def test_build_multiple2():
    dist1 = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    dist2 = ProbPoss({1: Fraction(1, 3), 2: Fraction(2, 3)})
    result = ProbPoss(
        {frozenset({1}): Fraction(1, 6), frozenset({1, 4}): Fraction(1, 2), frozenset({4}): Fraction(1, 3)}
    )
    a: Dict[PlayerName, ProbPoss[A]]
    a = {PlayerName("1"): dist1, PlayerName("2"): dist2}

    def f(x):
        return frozenset(valmap(lambda x: x ** 2, x).values())

    b = ProbabilityFraction()
    dist = b.build_multiple(a, f)
    eq_(dist, result)
