from dataclasses import dataclass

from dg_commons.sim import ImpactLocation

__all__ = ["SimpleCollision"]


@dataclass(frozen=True)
class SimpleCollision:
    location: ImpactLocation
    at_fault: bool
    energy_transfer: float
