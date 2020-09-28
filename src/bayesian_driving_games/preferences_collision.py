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
from .collisions import Collision

__all__ = ["CollisionPreference"]


class CollisionPreference(Preference[Optional[Collision]]):
    def __init__(self):
        self.p = SmallerPreferredTol(D(0))

    def get_type(self) -> Type[Optional[Collision]]:
        return Optional[Collision]

    def compare(self, a: Optional[Collision], b: Optional[Collision]) -> ComparisonOutcome:
        if a is None and b is None:
            return INDIFFERENT
        if a is None and b is not None:
            return FIRST_PREFERRED
        if a is not None and b is None:
            return SECOND_PREFERRED
        assert a is not None
        assert b is not None
        if a.active and not b.active:
            return SECOND_PREFERRED
        if b.active and not a.active:
            return FIRST_PREFERRED

        ea = a.energy_received + a.energy_transmitted
        eb = b.energy_received + b.energy_transmitted
        res = self.p.compare(ea, eb)
        assert res in COMP_OUTCOMES, (res, self.p)
        # logger.info('collision_pref', a=a, b=b, res=res)
        return res

    def __repr__(self) -> str:
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "CollisionPreference:\n " + debug_print(d)
