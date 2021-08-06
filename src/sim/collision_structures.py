from dataclasses import dataclass
from typing import NewType, Mapping, Sequence

import numpy as np
from shapely.geometry import Point, Polygon

from games import PlayerName
from sim import SimTime

__all__ = ["ImpactLocation",
           "IMPACT_FRONT",
           "IMPACT_BACK",
           "IMPACT_LEFT",
           "IMPACT_RIGHT",
           "CollisionReportPlayer",
           "CollisionReport",
           ]

ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_LEFT = ImpactLocation("left")
IMPACT_RIGHT = ImpactLocation('right')


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReportPlayer:
    locations: Sequence[ImpactLocation]
    """ Location of the impact """
    at_fault: bool
    """ At fault is defined as...."""
    footprint: Polygon
    """ Footprint of impact"""
    velocity: np.ndarray
    """ velocity before impact [m/s] """
    velocity_after: np.ndarray
    """ velocity after impact [m/s] """
    energy_delta: float
    """ Kinetic energy lost in the collision [J] """


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReport:
    players: Mapping[PlayerName, CollisionReportPlayer]
    """ Relative velocity defined as v_a-v_b in global RF [m/s] """
    impact_point: Point
    """Point of impact"""
    impact_normal: np.ndarray
    """Normal of impact"""
    at_time: SimTime
    """ Sim time at which the collision occurred"""
