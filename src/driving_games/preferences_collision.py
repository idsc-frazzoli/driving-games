from typing import Optional, Type

from zuper_typing import debug_print

from driving_games.collisions import SimpleCollision
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferred,
)

__all__ = ["CollisionPreference"]


class CollisionPreference(Preference[SimpleCollision]):
    def __init__(self):
        self.p = SmallerPreferred()

    def get_type(self) -> Type[SimpleCollision]:
        return SimpleCollision

    def compare(self, a: Optional[SimpleCollision], b: Optional[SimpleCollision]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None and b is not None:
            return FIRST_PREFERRED
        if a is not None and b is None:
            return SECOND_PREFERRED
        assert a is not None
        assert b is not None
        if a.at_fault and not b.at_fault:
            return SECOND_PREFERRED
        if b.at_fault and not a.at_fault:
            return FIRST_PREFERRED

        ea, eb = a.impact_rel_speed, b.impact_rel_speed
        res = self.p.compare(ea, eb)
        assert res in COMP_OUTCOMES, (res, self.p)
        return res

    def __repr__(self) -> str:
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "CollisionPreference:\n " + debug_print(d)
