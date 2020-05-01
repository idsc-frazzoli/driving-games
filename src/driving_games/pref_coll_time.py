from decimal import Decimal as D
from typing import Type

from games import Combined
from preferences import COMP_OUTCOMES, ComparisonOutcome, LexicographicPreference, Preference, SmallerPreferredTol
from zuper_typing import debug_print
from .collision_preference import CollisionPreference
from .structures import CollisionCost


class VehiclePreferencesCollTime(Preference[Combined[CollisionCost, D]]):
    def __init__(self, ignore_second=False):
        self.ignore_second = ignore_second
        self.collision = CollisionPreference()
        self.time = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.time))

    def get_type(self) -> Type[Combined[CollisionCost, D]]:
        return Combined[CollisionCost, D]

    def __repr__(self) -> str:
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(self, a: Combined[CollisionCost, D], b: Combined[CollisionCost, D]) -> ComparisonOutcome:
        # check_isinstance(a, Combined)
        # check_isinstance(b, Combined)
        if self.ignore_second:
            if a.joint is None and b.joint is None:
                return self.time.compare(a.personal, b.personal)
            else:
                return self.collision.compare(a.joint, b.joint)

        else:
            ct_a = (a.joint, a.personal)
            ct_b = (b.joint, b.personal)

            res = self.lexi.compare(ct_a, ct_b)
            assert res in COMP_OUTCOMES, (res, self.lexi)

            # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
            return res
