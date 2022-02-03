from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction

from dg_commons import SE2Transform
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import LightsCmd, NO_LIGHTS

__all__ = [
    "VehicleTimeCost",
    "VehicleSafetyDistCost",
    "VehicleTrackState",
    "VehicleActions",
]


@dataclass(frozen=True)
class VehicleTimeCost:
    """The personal costs of the vehicle"""

    __slots__ = ["duration"]
    duration: D
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleTimeCost":
        # weighting costs, e.g. according to a probability
        return replace(self, duration=self.duration * D(float(weight)))

    __rmul__ = __mul__

    # Monoid to support sum
    def __add__(self, other: "VehicleTimeCost") -> "VehicleTimeCost":
        if isinstance(other, VehicleTimeCost):
            return replace(self, duration=self.duration + other.duration)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__


@dataclass(frozen=True)
class VehicleSafetyDistCost:
    """The personal costs of the vehicle"""

    distance: float
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleSafetyDistCost":
        # weighting costs, e.g. according to a probability
        return replace(self, distance=self.distance * float(weight))

    __rmul__ = __mul__


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleTrackState:
    x: D
    """ Longitudinal progress """

    v: D
    """ Longitudinal velocity """

    wait: D
    """ How long we have been at speed = 0. We want to keep track so to bound this."""

    light: LightsCmd
    """ The current lights signal."""

    # todo maybe add has collided with other vehicle

    __print_order__ = ["x", "v"]  # only print these attribute

    def to_global_pose(self, ref_lane: DgLanelet) -> SE2Transform:
        beta = ref_lane.beta_from_along_lane(float(self.x))
        return SE2Transform.from_SE2(ref_lane.center_point(beta))


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    acc: D
    light: LightsCmd = NO_LIGHTS
