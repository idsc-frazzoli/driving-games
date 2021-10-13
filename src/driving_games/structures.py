from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from typing import Tuple

from dg_commons import Color
from dg_commons.sim.models.vehicle_ligths import LightsCmd, NO_LIGHTS

__all__ = [
    "VehicleCosts",
    "VehicleState",
    "VehicleActions",
    "VehicleGeometry",
    "SE2_disc",
]

SE2_disc = Tuple[D, D, D]  # in degrees


@dataclass(frozen=True)
class VehicleCosts:
    """The personal costs of the vehicle"""

    __slots__ = ["duration"]
    duration: D
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleCosts":
        # weighting costs, e.g. according to a probability
        return replace(self, duration=self.duration * D(float(weight)))

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleCosts") -> "VehicleCosts":
        if isinstance(other, VehicleCosts):
            return replace(self, duration=self.duration + other.duration)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__


@dataclass(frozen=True)
class VehicleGeometry:
    mass: D
    """ Mass [kg] """
    width: D
    """ Car width [m] """
    length: D
    """ Car length [m] """
    color: Color
    """ Car color """


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleState:
    ref: SE2_disc
    """ Reference frame from where the vehicle started """

    x: D
    """ Longitudinal position """

    v: D
    """ Longitudinal velocity """

    wait: D
    """ How long we have been at speed = 0. We want to keep track so bound this. """

    light: LightsCmd
    """ The current lights signal. """

    __print_order__ = ["x", "v"]  # only print these attributes


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    accel: D
    light: LightsCmd = NO_LIGHTS
