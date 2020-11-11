from possibilities.prob import enumerate_prob_assignments
from . import logger


def test_prob1():
    for i in [1, 2, 3, 4, 5]:
        res = enumerate_prob_assignments(i)
        logger.info(i=i, res=res)
