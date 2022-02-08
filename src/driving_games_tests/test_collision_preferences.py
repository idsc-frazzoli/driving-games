from math import pi
from typing import Mapping, Optional, Tuple

from nose.tools import assert_equal

from dg_commons import PlayerName
from driving_games import CollisionPreference, VehicleTimeCost, SimpleCollision
from driving_games import logger
from driving_games.zoo import get_asym
from games import Combined
from games.solve.solution_utils import get_outcome_preferences_for_players
from possibilities import PossibilityMonad, PossibilitySet
from preferences import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    SECOND_PREFERRED,
    StrictProductPreferenceDict,
)

P1 = PlayerName("p1")
P2 = PlayerName("p2")
P3 = PlayerName("p3")


def test1() -> None:
    C1 = SimpleCollision(0, False, rel_impact_direction=-pi / 5, impact_rel_speed=78)
    C2 = SimpleCollision(0, True, rel_impact_direction=pi / 5, impact_rel_speed=35)
    expect: Mapping[Tuple[Optional[SimpleCollision], Optional[SimpleCollision]], ComparisonOutcome]
    expect = {
        (None, None): INDIFFERENT,
        (C1, C1): INDIFFERENT,
        (None, C1): FIRST_PREFERRED,
        (C1, None): SECOND_PREFERRED,
        (C1, C2): FIRST_PREFERRED,
        (C2, C1): SECOND_PREFERRED,
    }
    pref = CollisionPreference()
    for (a, b), c in expect.items():
        res = pref.compare(a, b)
        assert_equal(res, c)


def test2() -> None:
    # * │ *         ╔═════════════════════════════════╗
    # > │     │ │   ║Outcome *                        ║
    # > │     │ │   ║│ private: {p1: Dec 3, p2: Dec 3}║
    # > │     │ │   ║│ joint:                         ║
    # > │     │ │   ║│ fdict                          ║
    # > │     │ │   ║│ │ p1: CollisionCost(v=Dec 1) * ║
    # > │     │ │   ║│ │ p2: CollisionCost(v=Dec 1) * ║
    # > │     │ │   ╚═════════════════════════════════╝
    # > │     │ │ fdict
    # > │     │ │ │ p1: {VehicleActions(accel=Dec 0) *}
    # > │     │ │ │ p2: {VehicleActions(accel=Dec 1) *}:
    # > │     │ │ fset
    # > │     │ │ * 'Outcome(private={p1: Dec 13, p2: Dec 4}, joint={}) *'
    ps: PossibilityMonad = PossibilitySet()

    c0 = SimpleCollision(
        0.0,
        True,
        5,
        3,
    )

    o_A = {
        P1: ps.unit(Combined(VehicleTimeCost(3), c0)),
        P2: ps.unit(Combined(VehicleTimeCost(3), c0)),
    }
    o_B = {
        P1: ps.unit(Combined(VehicleTimeCost(13), None)),
        P2: ps.unit(Combined(VehicleTimeCost(4), None)),
    }

    preferences = get_outcome_preferences_for_players(game)

    # preferences_ = tuple(preferences.values())
    eq_pref = StrictProductPreferenceDict(preferences)

    res = eq_pref.compare(o_A, o_B)
    logger.info(a=o_A, b=o_B, res=res)
    assert res == SECOND_PREFERRED, res


def test_3() -> None:
    game = get_asym().game
    p1, p2 = list(game.players)
    s1 = list(game.players[p1].initial.support())[0]
    sr = game.players[p1].dynamics.get_shared_resources(s1)
    logger.info(sr=sr)
