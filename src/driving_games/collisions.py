from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional

from zuper_commons.types import ZValueError

from dg_commons.sim import ImpactLocation

__all__ = ["SimpleCollision", "VehicleSafetyDistCost", "VehicleJointCost"]


@dataclass(frozen=True)
class SimpleCollision:
    location: ImpactLocation
    at_fault: bool
    energy_delta: float

    # ZZZ: this monoidal structure cna be controversial
    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "SimpleCollision":
        # weighting costs, e.g. according to a probability

        return replace(self, energy_delta=self.energy_delta * weight)

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "SimpleCollision") -> "SimpleCollision":
        if other is None:
            return self
        elif isinstance(other, SimpleCollision):
            return replace(
                self,
                ## impact location?!
                at_fault=self.at_fault or other.at_fault,
                energy_delta=self.energy_delta + other.energy_delta,
            )
        else:
            raise ZValueError("Cannot add a SimpleCollision to a non-SimpleCollision", other=type(other))

    __radd__ = __add__


@dataclass(frozen=True)
class VehicleSafetyDistCost:
    """Minimum safety distance costs of the vehicle"""

    distance: float
    """ Violation of the minimum safety distance """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleSafetyDistCost":
        # weighting costs, e.g. according to a probability
        return replace(self, distance=self.distance * float(weight))

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleSafetyDistCost") -> "VehicleSafetyDistCost":
        if isinstance(other, VehicleSafetyDistCost):
            return replace(self, distance=self.distance + other.distance)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__


@dataclass(frozen=True)
class VehicleJointCost:
    safety_dist: VehicleSafetyDistCost
    collision: Optional[SimpleCollision] = None

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleJointCost":
        # weighting costs, e.g. according to a probability
        return replace(self, safety_dist=self.safety_dist * weight, collision=self.collision * weight)

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleJointCost") -> "VehicleJointCost":
        if isinstance(other, VehicleJointCost):
            return replace(
                self, safety_dist=self.safety_dist + other.safety_dist, collision=self.collision + other.collision
            )
        else:
            raise ZValueError("Cannot add a VehicleJointCost to a non-VehicleJointCost", other=type(other))

    __radd__ = __add__
