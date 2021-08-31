from bisect import bisect_left, bisect_right

from games import PlayerName
from sim import SimLog


def test_bisect_simulationLog():
    # todo test logs performance
    log = SimLog()
    log[0] = {PlayerName("P1"): 1, PlayerName("P1"): 2}
    log[4] = {PlayerName("P1"): 4, PlayerName("P1"): 4}

    print("after ", log.get_entry_after(0))
    print("before ", log.get_entry_before(0))

    print(bisect_left(list(log.keys()), 0))
    print(bisect_right(list(log.keys()), 0))
