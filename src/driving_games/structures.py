from dataclasses import dataclass, replace
from decimal import Decimal as D
from fractions import Fraction
from functools import lru_cache

from dg_commons import SE2Transform
from dg_commons.maps import DgLanelet
from dg_commons.sim.models.vehicle_ligths import LightsCmd, NO_LIGHTS

__all__ = [
    "VehicleTimeCost",
    "VehicleTrackState",
    "VehicleActions",
]


@dataclass(frozen=True)
class VehicleTimeCost:
    """The personal costs of the vehicle"""

    duration: float
    """ Duration of the episode. """

    # support weight multiplication for expected value
    def __mul__(self, weight: Fraction) -> "VehicleTimeCost":
        # weighting costs, e.g. according to a probability
        return replace(self, duration=self.duration * weight)

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

    has_collided: bool
    """ Whether the vehicle has collided with something/someone. """

    __print_order__ = ["x", "v", "has_collided"]  # only print these attribute

    @lru_cache(maxsize=None)
    def to_global_pose(self, ref_lane: DgLanelet) -> SE2Transform:
        beta = ref_lane.beta_from_along_lane(float(self.x))
        return ref_lane.center_point_fast_SE2Transform(beta)

    # support weight multiplication for interpolation
    def __mul__(self, weight: float) -> "VehicleTrackState":
        # weighting costs, e.g. according to a probability
        return replace(self, x=self.x * D(weight), v=self.v * D(weight))

    __rmul__ = __mul__

    # Monoid to support sum for interpolation
    def __add__(self, other: "VehicleTrackState") -> "VehicleTrackState":
        if isinstance(other, VehicleTrackState):
            return replace(self, x=self.x + other.x, v=self.v + other.v)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    acc: D
    light: LightsCmd = NO_LIGHTS
