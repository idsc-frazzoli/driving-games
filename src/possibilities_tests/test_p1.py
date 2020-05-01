from nose.tools import assert_equal

from possibilities import PossibilityStructure, ProbabilityFraction, ProbabilitySet
from zuper_commons.types import ZValueError
from . import logger


def check_possibilities(ps: PossibilityStructure):
    a = ps.lift_one("a")
    logger.info(a=a)
    b = ps.lift_many({"A", "c"})
    logger.info(b=b)
    ab = ps.lift_many({a, b})
    logger.info(ab=ab)
    f = ps.flatten(ab)
    logger.info(f=f)
    g = ps.build(f, lambda _: _.upper())
    logger.info(g=g)

    sampler = ps.get_sampler(1)
    h = sampler.sample(g)
    logger.info(h=h)

    i = ps.mix({"a", "b"})
    logger.info(i=i)


def test_set():
    ps = ProbabilitySet()
    check_possibilities(ps)


def test_fraction():
    ps = ProbabilityFraction()
    check_possibilities(ps)


def test_set_mix():
    a = [0, 1]
    ps = ProbabilitySet()
    r = ps.mix(a)
    logger.info(r=r)
    assert_equal(r, {ps.lift_many({0}), ps.lift_many({1}), ps.lift_many({0, 1})})


def test_prob_mix():
    a = "a"
    b = "b"
    S = [a, b]
    ps = ProbabilityFraction()
    r = ps.mix(S)
    logger.info(r=r)
    # assert_equal(r, {ps.lift_many({0}), ps.lift_many({1}), ps.lift_many({0, 1})})


def test_prob_mix4():
    S = set("abcd")
    ps = ProbabilityFraction()
    r = ps.mix(S)
    if len(r) != 23:
        raise ZValueError(S=S, r=r, l=len(r))
