from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from typing import NewType

__all__ = ["ImpactLocation", "IMPACT_BACK", "IMPACT_FRONT", "IMPACT_SIDES", "Collision"]

ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_SIDES = ImpactLocation("sides")
# IMPACT_RIGHT = ImpactLocation('right')


@dataclass(frozen=True)
class Collision:
    # Where the impact was for this vehicle
    location: ImpactLocation
    active: bool
    """ Whether the car was active in the collision. Defined as: the collision
        would have occurred even if all the other cars were stopped. """
    # How much energy was received / transmitted
    energy_received: D
    energy_transmitted: D

    # Monoid for sum of Combined outcome
    # todo support for sum with None
    def __add__(self, other: "Collision") -> "Collision":
        if other is None:
            return self
        elif type(other) == type(self):
            return replace(
                self,
                energy_received=self.energy_received + other.energy_received,
                energy_transmitted=self.energy_transmitted + other.energy_transmitted,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "Collision":
        # weighting costs, e.g. according to a probability
        return replace(
            self,
            energy_received=self.energy_received * D(float(weight)),
            energy_transmitted=self.energy_transmitted * D(float(weight)),
        )

    __rmul__ = __mul__
    # todo what was this for? how location and similar shall be propagated?
    # def __add__(self, other: "Collision") -> "Collision":
    #     return replace(
    #         self,
    #         energy_received=self.energy_received + other.energy_received,
    #         energy_transmitted=self.energy_transmitted + other.energy_transmitted,
    #         location=self.location,
    #         active=self.active,
    #     )
