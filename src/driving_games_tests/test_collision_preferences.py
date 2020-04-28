from decimal import Decimal as D
from typing import Mapping, Optional, Tuple

from frozendict import frozendict
from nose.tools import assert_equal

from driving_games import CollisionCost, CollisionPreference, get_game1
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
    C1 = CollisionCost(D(1))
    C2 = CollisionCost(D(2))
    expect: Mapping[Tuple[Optional[CollisionCost], Optional[CollisionCost]], ComparisonOutcome]
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
    o_A = Outcome(
        private=frozendict({p1: D(3), p2: D(3)}),
        joint=frozendict({p1: CollisionCost(D(1)), p2: CollisionCost(D(1))}),
    )
    o_B = Outcome(private=frozendict({p1: D(13), p2: D(4)}), joint=frozendict())

    outcomes_A = frozenset({o_A})
    outcomes_B = frozenset({o_B})

    game = get_game1()
    preferences = get_outcome_set_preferences_for_players(game)

    preferences_ = tuple(preferences.values())
    eq_pref = StrictProductPreference(preferences_)

    res = eq_pref.compare(outcomes_A, outcomes_B)
    logger.info(a=outcomes_A, b=outcomes_B, res=res)
    assert res == SECOND_PREFERRED
