from decimal import Decimal as D
from typing import Mapping, Optional, Tuple

from frozendict import frozendict
from nose.tools import assert_equal

from driving_games import Collision, CollisionPreference, get_game1
from driving_games.collisions import IMPACT_FRONT
from games import get_outcome_set_preferences_for_players, Outcome
from preferences import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    SECOND_PREFERRED,
    StrictProductPreference,
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
    p1 = "p1"
    p2 = "p2"
    c0 = Collision(IMPACT_FRONT, True, D(1), D(0))
    o_A = Outcome(private=frozendict({p1: D(3), p2: D(3)}), joint=frozendict({p1: c0, p2: c0}),)
    o_B = Outcome(private=frozendict({p1: D(13), p2: D(4)}), joint=frozendict())

    game = get_game1()

    outcomes_A = game.ps.lift_one(o_A)
    outcomes_B = game.ps.lift_one(o_B)

    preferences = get_outcome_set_preferences_for_players(game)

    preferences_ = tuple(preferences.values())
    eq_pref = StrictProductPreference(preferences_)

    res = eq_pref.compare(outcomes_A, outcomes_B)
    logger.info(a=outcomes_A, b=outcomes_B, res=res)
    assert res == SECOND_PREFERRED, res
