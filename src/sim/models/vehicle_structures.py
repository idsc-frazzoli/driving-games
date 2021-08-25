from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, NewType

import numpy as np
from geometry import SE2_from_xytheta

from sim.models.model_structures import ModelGeometry

__all__ = ["VehicleType", "CAR", "MOTORCYCLE", "BICYCLE", "VehicleGeometry"]

VehicleType = NewType("VehicleType", str)
CAR = VehicleType("car")
MOTORCYCLE = VehicleType("motorcycle")
BICYCLE = VehicleType("bicycle")


@dataclass(frozen=True, unsafe_hash=True)
class VehicleGeometry(ModelGeometry):
    """ Geometry parameters of the vehicle (and colour)"""

    vehicle_type: VehicleType
    """Type of the vehicle"""
    w_half: float
    """ Half width of vehicle [m] """
    lf: float
    """ Front length of vehicle - dist from CoG to front [m] """
    lr: float
    """ Rear length of vehicle - dist from CoG to back [m] """
    h_cog: float = 0.7
    """ Height of the CoG [m] """

    # todo fix default rotational inertia
    @classmethod
    def default_car(cls) -> "VehicleGeometry":
        return VehicleGeometry(vehicle_type=CAR, m=1500.0, Iz=1000, w_half=.95, lf=1.95, lr=1.95, e=0.6,
                               color="royalblue")

    @classmethod
    def default_bicycle(cls) -> "VehicleGeometry":
        return VehicleGeometry(vehicle_type=BICYCLE, m=80.0, Iz=80, w_half=0.25, lf=1.0, lr=1.0, e=0.3,
                               color="saddlebrown")

    @cached_property
    def width(self):
        return self.w_half * 2

    @cached_property
    def length(self):
        return self.lf + self.lr

    @cached_property
    def outline(self) -> Tuple[Tuple[float, float], ...]:
        return ((-self.lr, -self.w_half), (-self.lr, +self.w_half),
                (+self.lf, +self.w_half), (+self.lf, -self.w_half), (-self.lr, -self.w_half))

    @cached_property
    def wheel_shape(self):
        if self.vehicle_type == CAR:
            halfwidth, radius = 0.1, 0.3  # size of the wheels
        elif self.vehicle_type == MOTORCYCLE or self.vehicle_type == BICYCLE:
            halfwidth, radius = 0.05, 0.3  # size of the wheels
        else:
            raise ValueError("Unrecognised vehicle type while trying to get weels outline")
        return halfwidth, radius

    @cached_property
    def wheel_outline(self):
        halfwidth, radius = self.wheel_shape
        # fixme uniform points handlings to native list of tuples
        return np.array([[radius, -radius, -radius, radius, radius],
                         [-halfwidth, -halfwidth, halfwidth, halfwidth, -halfwidth],
                         [1, 1, 1, 1, 1]])

    @cached_property
    def wheels_position(self) -> np.ndarray:
        halfwidth, radius = self.wheel_shape
        if self.vehicle_type == CAR:
            backwardshift = self.lf / 4
            # return 4 wheels position (always the first half are the front ones)
            positions = np.array(
                [[self.lf - radius - backwardshift, self.lf - radius - backwardshift, -self.lr + radius,
                  -self.lr + radius],
                 [self.w_half - halfwidth, -self.w_half + halfwidth, self.w_half - halfwidth,
                  -self.w_half + halfwidth], [1, 1, 1, 1]])

        else:  # self.vehicle_type == MOTORCYCLE or self.vehicle_type == BICYCLE
            positions = np.array([[self.lf - radius, -self.lr + radius], [0, 0], [1, 1]])
        return positions

    @cached_property
    def n_wheels(self) -> int:
        return self.wheels_position.shape[1]

    def get_rotated_wheels_outlines(self, delta: float) -> List[np.ndarray]:
        """
        :param delta: Steering angle of front wheels
        :return:
        """
        wheels_position = self.wheels_position
        assert self.n_wheels in (2, 4), self.n_wheels
        transformed_wheels_outlines = []
        for i in range(self.n_wheels):
            # the first half of the wheels are the ones that get rotated
            if i < self.n_wheels / 2:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], delta))
            else:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], 0))
            transformed_wheels_outlines.append(transform @ self.wheel_outline)
        return transformed_wheels_outlines
