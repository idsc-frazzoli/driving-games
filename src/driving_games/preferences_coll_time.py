from decimal import Decimal as D
from typing import Type

from zuper_typing import debug_print

from driving_games.collisions import VehicleJointCost
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


class VehiclePreferencesCollTime(Preference[Combined[VehicleJointCost, VehicleTimeCost]]):
    def __init__(self, ignore_second=False):
        self.ignore_second = ignore_second
        self.collision = CollisionPreference()
        self.scalar_pref = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.scalar_pref, self.scalar_pref))

    def get_type(self) -> Type[Combined[VehicleJointCost, D]]:
        return Combined[VehicleJointCost, VehicleTimeCost]

    def __repr__(self) -> str:
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[VehicleJointCost, VehicleTimeCost], b: Combined[VehicleJointCost, VehicleTimeCost]
    ) -> ComparisonOutcome:
        ct_a = (a.joint.collision, a.joint.safety_dist_violation.distance, a.personal.duration)
        ct_b = (b.joint.collision, b.joint.safety_dist_violation.distance, b.personal.duration)

        res = self.lexi.compare(ct_a, ct_b)
        assert res in COMP_OUTCOMES, (res, self.lexi)
        # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
        return res
