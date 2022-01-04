from decimal import Decimal as D
from typing import Type

from zuper_typing import debug_print

from dg_commons.sim import CollisionReportPlayer
from driving_games.preferences_collision import CollisionPreference
from driving_games.structures import VehicleTimeCost
from games import Combined
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    LexicographicPreference,
    Preference,
    SmallerPreferredTol,
)

__all__ = ["VehiclePreferencesCollTime"]


class VehiclePreferencesCollTime(Preference[Combined[CollisionReportPlayer, VehicleTimeCost]]):
    def __init__(self, ignore_second=False):
        self.ignore_second = ignore_second
        self.collision = CollisionPreference()
        self.scalar_pref = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.scalar_pref))

    def get_type(self) -> Type[Combined[CollisionReportPlayer, D]]:
        return Combined[CollisionReportPlayer, VehicleTimeCost]

    def __repr__(self) -> str:
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[CollisionReportPlayer, VehicleTimeCost], b: Combined[CollisionReportPlayer, VehicleTimeCost]
    ) -> ComparisonOutcome:
        if self.ignore_second:
            if a.joint is None and b.joint is None:
                return self.scalar_pref.compare(a.personal.duration, b.personal.duration)
            else:
                return self.collision.compare(a.joint, b.joint)
        else:
            ct_a = (a.joint, a.personal.duration)
            ct_b = (b.joint, b.personal.duration)

            res = self.lexi.compare(ct_a, ct_b)
            assert res in COMP_OUTCOMES, (res, self.lexi)

            # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
            return res
