from decimal import Decimal as D
from typing import Optional, Type

from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferredTol,
)
from zuper_typing import debug_print
from .structures import CollisionCost

__all__ = ["CollisionPreference"]


class CollisionPreference(Preference[Optional[CollisionCost]]):
    def __init__(self):
        self.p = SmallerPreferredTol(D(0))

    def get_type(self) -> Type[Optional[CollisionCost]]:
        return Optional[CollisionCost]

    def compare(self, a: Optional[CollisionCost], b: Optional[CollisionCost]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None and b is not None:
            return FIRST_PREFERRED
        if a is not None and b is None:
            return SECOND_PREFERRED
        assert a is not None
        assert b is not None
        res = self.p.compare(a.v, b.v)
        assert res in COMP_OUTCOMES, (res, self.p)
        # logger.info('collision_pref', a=a, b=b, res=res)
        return res

    def __repr__(self) -> str:
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "CollisionPreference:\n " + debug_print(d)
