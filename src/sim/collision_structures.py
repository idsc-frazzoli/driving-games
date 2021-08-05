from dataclasses import dataclass
from typing import NewType, List

import numpy as np

from sim import SimTime

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
    rel_velocity: np.ndarray
    """ Relative velocity defined as v_a-v_b in global RF [m/s] """
    # rel_velocity_along_n: float
    """ Relative velocity defined as dot(v_a,n)-dot(v_b,n), with n the normal of impact """
    energy_delta: float
    """ Kinetic energy lost in the collision [J] """
    at_time: SimTime
    """ Sim time at which the collision occurred"""
