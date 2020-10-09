from nose.tools import assert_equal

from possibilities import PossibilityMonad, PossibilitySet

from . import logger


def check_possibilities(ps: PossibilityMonad):
    a = ps.unit("a")
    logger.info(a=a)
    b = ps.lift_many({"A", "c"})
    logger.info(b=b)
    ab = ps.lift_many({a, b})
    logger.info(ab=ab)
    f = ps.join(ab)
    logger.info(f=f)
    g = ps.build(f, lambda _: _.upper())
    logger.info(g=g)

    sampler = ps.get_sampler(1)
    h = sampler.sample(g)
    logger.info(h=h)

    i = ps.mix({"a", "b"})
    logger.info(i=i)


def test_set():
    ps = PossibilitySet()
    check_possibilities(ps)


def test_set_mix():
    a = [0, 1]
    ps = PossibilitySet()
    r = ps.mix(a)
    logger.info(r=r)
    assert_equal(r, {ps.lift_many({0}), ps.lift_many({1}), ps.lift_many({0, 1})})
