from decimal import Decimal as D
from typing import Type

from games import Combined
from preferences import (
    COMP_OUTCOMES,
    ComparisonOutcome,
    LexicographicPreference,
    Preference,
    SmallerPreferredTol,
)
from zuper_typing import debug_print
from driving_games.collisions import Collision
from driving_games.preferences_collision import CollisionPreference
from driving_games.structures import VehicleCosts

__all__ = ["VehiclePreferencesCollTime"]


class VehiclePreferencesCollTime(Preference[Combined[Collision, VehicleCosts]]):
    def __init__(self, ignore_second=False):
        self.ignore_second = ignore_second
        self.collision = CollisionPreference()
        self.time = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.time))

    def get_type(self) -> Type[Combined[Collision, D]]:
        return Combined[Collision, VehicleCosts]

    def __repr__(self) -> str:
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[Collision, VehicleCosts], b: Combined[Collision, VehicleCosts]
    ) -> ComparisonOutcome:
        # check_isinstance(a, Combined)
        # check_isinstance(b, Combined)
        if self.ignore_second:
            if a.joint is None and b.joint is None:

                return self.time.compare(a.personal.duration, b.personal.duration)
            else:
                return self.collision.compare(a.joint, b.joint)

        else:
            ct_a = (a.joint, a.personal.duration)
            ct_b = (b.joint, b.personal.duration)

            res = self.lexi.compare(ct_a, ct_b)
            assert res in COMP_OUTCOMES, (res, self.lexi)

            # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
            return res

class VehiclePreferencesCollTimeML(Preference[Combined[Collision, VehicleCosts]]):
    def __init__(self, ignore_second=True):
        self.ignore_second = ignore_second
        self.collision = CollisionPreference()
        self.time = SmallerPreferredTol(D(0))
        self.lexi = LexicographicPreference((self.collision, self.time))

    def get_type(self) -> Type[Combined[Collision, D]]:
        return Combined[Collision, VehicleCosts]

    def __repr__(self) -> str:
        d = {"P": self.get_type(), "lexi": self.lexi}
        return "VehiclePreferencesCollTime: " + debug_print(d)

    def compare(
        self, a: Combined[Collision, VehicleCosts], b: Combined[Collision, VehicleCosts]
    ) -> ComparisonOutcome:
        # check_isinstance(a, Combined)
        # check_isinstance(b, Combined)
        if self.ignore_second:
            if a.joint is None and b.joint is None:

                return self.time.compare(a.personal.duration, b.personal.duration)
            else:
                return self.collision.compare(a.joint, b.joint)

        else:
            ct_a = (a.joint, a.personal.duration)
            ct_b = (b.joint, b.personal.duration)

            res = self.lexi.compare(ct_a, ct_b)
            assert res in COMP_OUTCOMES, (res, self.lexi)

            # logger.info(ct_a=ct_a, ct_b=ct_b, res=res)
            return res
