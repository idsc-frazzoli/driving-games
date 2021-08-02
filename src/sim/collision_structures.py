from dataclasses import dataclass
from typing import NewType, List

#from sim import SimTime

__all__ = ["ImpactLocation",
           "IMPACT_FRONT",
           "IMPACT_BACK",
           "IMPACT_LEFT",
           "IMPACT_RIGHT",
           "CollisionReport"
           ]

ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_LEFT = ImpactLocation("left")
IMPACT_RIGHT = ImpactLocation('right')


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReport:
    location: List[ImpactLocation]
    """ Location of the impact """
    at_fault: bool
    """ At fault is defined as...."""
    rel_velocity: float
    """ Relative velocity defined as... """
    energy_transfer: float
    """ """
    at_time: float #SimTime


