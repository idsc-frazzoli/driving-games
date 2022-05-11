from nose.tools import assert_almost_equal

from dg_commons import PlayerName, SE2Transform
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from trajectory_games.metrics import Clearance


def test_clearance():
    geo1 = VehicleGeometry.default_car(color=(1, 0, 0))
    p1 = PlayerName("P1")
    p2 = PlayerName("P2")
    pos0 = SE2Transform(p=[0.0, 0.0], theta=1.57)
    pos1 = SE2Transform(p=[1.0, 0.0], theta=1.57)
    pos2 = SE2Transform(p=[2.0, 0.0], theta=1.57)
    pos3 = SE2Transform(p=[2.1, 0.0], theta=1.57)
    pos4 = SE2Transform(p=[0.0, 2.0], theta=1.57)
    pos5 = SE2Transform(p=[0.0, 4.1], theta=1.57)
    pos6 = SE2Transform(p=[1.0, 0.0], theta=0.0)
    pos7 = SE2Transform(p=[1.0, 1.0], theta=1.0)
    pos8 = SE2Transform(p=[2.1, 4.1], theta=1.57)
    pos9 = SE2Transform(p=[3.1, 3.1], theta=0.0)
    pos10 = SE2Transform(p=[3.1, 3.1], theta=3.14)
    pos11 = SE2Transform(p=[-3.1, 3.1], theta=0.0)
    pos12 = SE2Transform(p=[3.1, -3.1], theta=3.14)
    pos13 = SE2Transform(p=[1.0, 1.0], theta=1.0)
    pos14 = SE2Transform(p=[-2.0, 1.0], theta=1.57)

    def calc(pos_p1: SE2Transform, pos_p2: SE2Transform) -> float:
        inp = {p1: (pos_p1, geo1), p2: (pos_p2, geo1)}
        return Clearance.get_clearance(inp)

    assert_almost_equal(calc(pos0, pos1), 0.0)
    assert_almost_equal(calc(pos0, pos2), 0.0)
    assert_almost_equal(calc(pos0, pos3), 0.1, places=2)
    assert_almost_equal(calc(pos0, pos4), 0.0)
    assert_almost_equal(calc(pos0, pos5), 0.1, places=2)
    assert_almost_equal(calc(pos0, pos6), 0.0)
    assert_almost_equal(calc(pos0, pos7), 0.0)
    assert_almost_equal(calc(pos0, pos8), 0.14, places=2)
    assert_almost_equal(calc(pos0, pos9), 0.14, places=2)
    assert_almost_equal(calc(pos0, pos10), 0.14, places=2)
    assert_almost_equal(calc(pos0, pos11), 0.14, places=2)
    assert_almost_equal(calc(pos0, pos12), 0.14, places=2)

    assert_almost_equal(calc(pos1, pos0), 0.0)
    assert_almost_equal(calc(pos2, pos0), 0.0)
    assert_almost_equal(calc(pos3, pos0), 0.1, places=2)
    assert_almost_equal(calc(pos4, pos0), 0.0)
    assert_almost_equal(calc(pos5, pos0), 0.1, places=2)
    assert_almost_equal(calc(pos6, pos0), 0.0)
    assert_almost_equal(calc(pos7, pos0), 0.0)
    assert_almost_equal(calc(pos8, pos0), 0.14, places=2)
    assert_almost_equal(calc(pos9, pos0), 0.14, places=2)
    assert_almost_equal(calc(pos10, pos0), 0.14, places=2)
    assert_almost_equal(calc(pos11, pos0), 0.14, places=2)
    assert_almost_equal(calc(pos12, pos0), 0.14, places=2)

    assert_almost_equal(calc(pos13, pos14), 0.079, places=2)
    assert_almost_equal(calc(pos14, pos13), 0.079, places=2)


if __name__ == "__main__":
    test_clearance()
