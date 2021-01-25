from typing import Union, Tuple
from dataclasses import dataclass
from decimal import Decimal as D
import numpy as np

import geometry as geo

from driving_games.structures import VehicleActions, VehicleCosts, SE2_disc, Lights

from duckie_games.utils import (
    interpolate_along_lane,
    from_SE2Transform_to_SE2_disc,
    get_SE2disc_from_along_lane,
    get_SE2disc_in_ref_from_along_lane,
    DuckietownMapHashable,
    LaneSegmentHashable
)


@dataclass(frozen=True)
class DuckieCosts(VehicleCosts):
    pass


@dataclass(frozen=True)
class DuckieGeometry:
    mass: D
    """ Mass [kg] """
    width: D
    """ Duckie width [m] """
    length: D
    """ Duckie length [m] """
    color: Union[str, Tuple[float, float, float]]
    """ Color of Duckiebot, e.g. "red", "green", "blue" """
    height: D
    """ Duckie Height [m] """


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieState:
    ref: SE2_disc
    """ Reference frame from where the vehicle started """

    lane: LaneSegmentHashable
    """ Lane that the duckie follows"""

    x: D
    """ Position along lane """

    v: D
    """ Velocity along lane """

    wait: D
    """ How long we have been at speed = 0. We want to keep track so bound this. """

    light: Lights
    """ The current lights signal. """

    __print_order__ = ["x", "v"]  # only print these attributes

    @property
    def abs_pose(self) -> SE2_disc:
        """
        get the absolute pose of the duckie in the map
        """
        return get_SE2disc_from_along_lane(lane=self.lane, along_lane=self.x)

    @property
    def ref_pose(self) -> SE2_disc:
        """
        get the pose of the duckie relative to the reference frame
        """
        return get_SE2disc_in_ref_from_along_lane(ref=self.ref, lane=self.lane, along_lane=self.x)


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieActions(VehicleActions):
    pass
