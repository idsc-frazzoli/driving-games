from typing import Union, Tuple
from dataclasses import dataclass
from decimal import Decimal as D
import numpy as np

import geometry as geo

from driving_games.structures import VehicleActions, VehicleCosts, SE2_disc, Lights

from duckie_games.utils import (
    interpolate_along_lane,
    from_SE2Transform_to_SE2_disc,
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
    duckie_map: DuckietownMapHashable
    """ Duckietown world map where the duckie is playing """

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
        """ get the absolute pose of the duckie in the map """

        pose_SE2_transform = interpolate_along_lane(lane=self.lane, along_lane=float(self.x))
        return from_SE2Transform_to_SE2_disc(pose_SE2_transform)

    @property
    def ref_pose(self) -> SE2_disc:
        """ get the pose of the duckie relative to the reference frame"""

        # Get the SE2 representation of the absolute pose
        *t_abs, theta_abs_deg = map(float, self.abs_pose)
        theta_abs_rad = np.deg2rad(theta_abs_deg)
        q_abs = geo.SE2_from_translation_angle(t_abs, theta_abs_rad)

        # Get SE2 representation of the ref pose
        *t_ref, theta_ref_deg = map(float, self.ref)
        theta_ref_rad = np.deg2rad(theta_ref_deg)
        q_ref = geo.SE2_from_translation_angle(t_ref, theta_ref_rad)

        # Get the the pose of the duckie in the reference frame
        q_abs_from_q_ref = geo.SE2.multiply(geo.SE2.inverse(q_ref), q_abs)
        t, theta_rad = geo.translation_angle_from_SE2(q_abs_from_q_ref)
        x, y = t
        theta_deg = np.rad2deg(theta_rad)
        return (D(x), D(y), D(theta_deg))


@dataclass(frozen=True, unsafe_hash=True, eq=True, order=True)
class DuckieActions(VehicleActions):
    pass
