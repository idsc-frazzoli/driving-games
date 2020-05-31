from decimal import Decimal as D
from typing import Mapping, Optional, Tuple

from nose.tools import assert_equal

from driving_games import (
    Collision,
    CollisionPreference,
    get_asym,
    IMPACT_FRONT,
    VehicleCosts,
)
from games import Combined, get_outcome_set_preferences_for_players
from preferences import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    SECOND_PREFERRED,
    StrictProductPreferenceDict,
)
from . import logger


def test1() -> None:
    C1 = Collision(IMPACT_FRONT, True, D(1), D(0))
    C2 = Collision(IMPACT_FRONT, True, D(2), D(0))
    expect: Mapping[Tuple[Optional[Collision], Optional[Collision]], ComparisonOutcome]
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
    # *│ * ╔═════════════════════════════════╗
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
    game = get_asym().game
    p1, p2 = list(game.players)
    c0 = Collision(IMPACT_FRONT, True, D(1), D(0))

    o_A = {
        p1: game.ps.lift_one(Combined(VehicleCosts(D(3)), c0)),
        p2: game.ps.lift_one(Combined(VehicleCosts(D(3)), c0)),
    }
    o_B = {
        p1: game.ps.lift_one(Combined(VehicleCosts(D(13)), None)),
        p2: game.ps.lift_one(Combined(VehicleCosts(D(4)), None)),
    }

    preferences = get_outcome_set_preferences_for_players(game)

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
