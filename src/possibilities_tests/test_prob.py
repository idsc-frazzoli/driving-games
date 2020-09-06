from fractions import Fraction
from typing import Mapping, Dict

from games import PlayerName, U, JointPureActions
from possibilities import Poss
from possibilities.prob import enumerate_prob_assignments, ProbabilityFraction, ProbPoss, A
from . import logger



def test_prob1():
    for i in [1, 2, 3, 4, 5]:
        res = enumerate_prob_assignments(i)
        logger.info(i=i, res=res)

def test_prob2():
    distribution = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    P1 = PlayerName('1')
    a: Dict[PlayerName, ProbPoss[A]]
    a = {P1: distribution}

    def f(x):
        x_squared = {key: pow(value, 2.) for key, value in x.items()}
        return list(x_squared.values())

    b = ProbabilityFraction()
    dist: Poss[JointPureActions] = b.build_multiple(a, f)
    print(dist)


def test_prob3():
    distribution1 = ProbPoss({1: Fraction(1, 2), 2: Fraction(1, 2)})
    distribution2 = ProbPoss({1: Fraction(1, 3), 2: Fraction(2, 3)})
    P1 = PlayerName('1')
    P2 = PlayerName('2')
    a: Dict[PlayerName, ProbPoss[A]]
    a = {P1: distribution1, P2: distribution2}

    def f(x):
        x_squared = {key: pow(value, 2.) for key, value in x.items()}
        return list(x_squared.values())

    b = ProbabilityFraction()
    dist: Poss[JointPureActions] = b.build_multiple(a, f)
    print(dist)