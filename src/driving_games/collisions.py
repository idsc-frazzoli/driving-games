from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional

from zuper_commons.types import ZValueError

from dg_commons import Timestamp
from driving_games import logger

__all__ = ["SimpleCollision", "VehicleSafetyDistCost", "VehicleJointCost"]


@dataclass(unsafe_hash=True, frozen=True)
class SimpleCollision:
    at: Timestamp
    """When the collision happened."""
    at_fault: bool
    """Whether the the player was at fault."""
    rel_impact_direction: float
    """Relative impact direction in polar coordinates (0 is the front, pi is the back)"""
    impact_rel_speed: float
    """Relative speed at impact"""

    # ZZZ: this monoidal structure cna be controversial
    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "SimpleCollision":
        # weighting costs, e.g. according to a probability

        return replace(self, impact_rel_speed=self.impact_rel_speed * weight)

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "SimpleCollision") -> "SimpleCollision":
        if other is None:
            return self
        elif isinstance(other, SimpleCollision):
            if self.at < other.at:
                return self
            elif self.at > other.at:
                return other
            else:
                logger.warning(f"Performing sum between SimpleCollision occurred at the same instant. Monoid?")
                return replace(  # monoid?
                    self,
                    at_fault=self.at_fault or other.at_fault,
                    impact_rel_speed=self.impact_rel_speed + other.impact_rel_speed,
                )
        else:
            raise ZValueError("Cannot add a SimpleCollision to a non-SimpleCollision", other=type(other))

    __radd__ = __add__


@dataclass(unsafe_hash=True, frozen=True)
class VehicleSafetyDistCost:
    """Minimum safety distance costs of the vehicle"""

    distance: float
    """ Violation of the minimum safety distance """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleSafetyDistCost":
        # weighting costs, e.g. according to a probability
        return replace(self, distance=self.distance * float(weight))

    __rmul__ = __mul__

    # Cost monoid to support sum
    def __add__(self, other: "VehicleSafetyDistCost") -> "VehicleSafetyDistCost":
        if isinstance(other, VehicleSafetyDistCost):
            return replace(self, distance=self.distance + other.distance)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    # def __neg__(self) -> "VehicleSafetyDistCost":
    #     return replace(self, distance=-self.distance)
    #
    # def __sub__(self, other: "VehicleSafetyDistCost"):
    #     return self + (-other)


@dataclass(unsafe_hash=True, frozen=True)
class VehicleJointCost:
    safety_dist_violation: VehicleSafetyDistCost
    collision: Optional[SimpleCollision] = None

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleJointCost":
        # weighting costs, e.g. according to a probability
        return replace(
            self, safety_dist_violation=self.safety_dist_violation * weight, collision=self.collision * weight
        )

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleJointCost") -> "VehicleJointCost":
        if isinstance(other, VehicleJointCost):
            return replace(
                self,
                safety_dist_violation=self.safety_dist_violation + other.safety_dist_violation,
                collision=self.collision if other.collision is None else self.collision + other.collision,
            )
        else:
            raise ZValueError("Cannot add a VehicleJointCost to a non-VehicleJointCost", other=type(other))

    __radd__ = __add__
