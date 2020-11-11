from dataclasses import dataclass
from decimal import Decimal as D
from typing import NewType

from .rectangle import Coordinates, Rectangle

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


@dataclass(frozen=True)
class ProjectedCar:
    rectangle: Rectangle
    front_left: Coordinates
    front_center: Coordinates
    front_right: Coordinates
