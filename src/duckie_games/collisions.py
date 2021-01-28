from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from typing import NewType, Optional

from duckie_games.rectangle import Angle_Deg


__all__ = [
    "ImpactLocation",
    "IMPACT_BACK",
    "IMPACT_FRONT",
    "IMPACT_SIDES",
    "IMPACT_RIGHT",
    "IMPACT_LEFT",
    "IMPACT_NONE",
    "Collision"
]


ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_SIDES = ImpactLocation("sides")
IMPACT_RIGHT = ImpactLocation('right')
IMPACT_LEFT = ImpactLocation('left')
IMPACT_NONE = ImpactLocation('none')


@dataclass(frozen=True)
class Collision:
    __slots__ = ["location", "angle", "active", "energy_received", "energy_transmitted"]
    location: ImpactLocation
    """Where the impact was for this vehicle"""
    angle: Optional[Angle_Deg]
    """Angle of impact in degrees"""
    active: bool
    """ Whether the car was active in the collision. Defined as: the collision
        would have occurred even if all the other cars were stopped. """
    energy_received: D
    """How much energy was received"""
    energy_transmitted: D
    """How much energy was transmitted"""

    # Monoid sum of Collision
    def __add__(self, other: "Collision") -> "Collision":
        if other is None:
            return self
        elif isinstance(other, Collision):
            # fixme how to propagate "active" and "location" ?
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
        w = D(float(weight))
        return replace(
            self, energy_received=self.energy_received * w, energy_transmitted=self.energy_transmitted * w
        )

    __rmul__ = __mul__
