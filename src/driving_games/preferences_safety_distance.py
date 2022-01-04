from typing import Type

from zuper_typing import debug_print

from driving_games.structures import VehicleSafetyDistCost
from preferences import (
    ComparisonOutcome,
    FIRST_PREFERRED,
    INDIFFERENT,
    Preference,
    SECOND_PREFERRED,
    SmallerPreferred,
)

__all__ = ["SafetyDistPreference"]


class SafetyDistPreference(Preference[VehicleSafetyDistCost]):
    def __init__(self):
        self.p = SmallerPreferred()

    def get_type(self) -> Type[VehicleSafetyDistCost]:
        return VehicleSafetyDistCost

    def compare(self, a: VehicleSafetyDistCost, b: VehicleSafetyDistCost) -> ComparisonOutcome:
        assert a is not None
        assert b is not None
        if a.distance < b.distance:
            return SECOND_PREFERRED
        if b.distance > a.distance:
            return FIRST_PREFERRED
        assert a.distance == b.distance
        return INDIFFERENT

    def __repr__(self) -> str:
        d = {
            "T": self.get_type(),
            "p": self.p,
        }
        return "SafetyDistPreference:\n " + debug_print(d)
