from numpy.testing import assert_raises

from games import PlayerName
from sim import SimLog


def test_bisect_simulationLog():
    # todo test logs performance
    log = SimLog()

    def tryillegal():
        log[0] = {PlayerName("P1"): 1, PlayerName("P1"): 2}

    log[0] = {PlayerName("P1"): 1, PlayerName("P1"): 2}
    #assert_raises(AssertionError, tryillegal)
