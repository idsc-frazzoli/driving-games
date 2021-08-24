# import math
# from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, NewType

import numpy as np
from geometry import SE2_from_xytheta

# from sim.models.utils import kmh2ms
# from sim.typing import Color
from sim.models.model_structures import ModelGeometry

__all__ = ["VehicleType", "CAR", "MOTORCYCLE", "BICYCLE", "VehicleGeometry"]
VehicleType = NewType("VehicleType", str)
CAR = VehicleType("car")
MOTORCYCLE = VehicleType("motorcycle")
BICYCLE = VehicleType("bicycle")

#
# @dataclass(frozen=True, unsafe_hash=True)
# class ModelGeometry(ABC):
#     m: float
#     """ Vehicle Mass [kg] """
#     Iz: float
#     """ Moment of inertia (used only in the dynamic model) """
#     e: float
#     """ Restitution coefficient (used only in collisions energy transfer).
#     Ratio of the differences in vehicle speeds before and after the collision -> 0 < e < 1"""
#
#     @property
#     @abstractmethod
#     def outline(self) -> Tuple[Tuple[float, float], ...]:
#         pass


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
    c_drag: float
    """ Drag coefficient """
    a_drag: float
    """ Section Area interested by drag """
    h_cog: float = 0.7
    """ Height of the CoG [m] """
    rho: float = 1.225
    """ air density """
    # color: Color = (1, 1, 1)
    # """ Color """

    # todo fix default rotational inertia
    @classmethod
    def default_car(cls) -> "VehicleGeometry":
        return VehicleGeometry(vehicle_type=CAR, m=1500.0, Iz=1000, w_half=1.0, lf=2.0, lr=2.0, c_drag=0.3756,
                               a_drag=2, e=0.6, color="royalblue")

    @classmethod
    def default_bicycle(cls) -> "VehicleGeometry":
        return VehicleGeometry(vehicle_type=BICYCLE, m=80.0, Iz=80, w_half=0.25, lf=1.0, lr=1.0, c_drag=0.01,
                               a_drag=0.2, e=0.5, color="saddlebrown")

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
#
#
# @dataclass(frozen=True, unsafe_hash=True)
# class VehicleParameters:
#     vx_limits: Tuple[float, float]
#     """ Minimum and Maximum velocities [m/s] """
#     acc_limits: Tuple[float, float]
#     """ Minimum and Maximum acceleration [m/s^2] """
#     delta_max: float
#     """ Maximum steering angle [rad] """
#     ddelta_max: float
#     """ Minimum and Maximum steering rate [rad/s] """
#
#     @classmethod
#     def default_car(cls) -> "VehicleParameters":
#         # data from https://copradar.com/chapts/references/acceleration.html
#         return VehicleParameters(vx_limits=(kmh2ms(-10), kmh2ms(130)),
#                                  acc_limits=(-8, 5),
#                                  delta_max=math.pi / 6,
#                                  ddelta_max=1)
#
#     @classmethod
#     def default_bicycle(cls) -> "VehicleParameters":
#         return VehicleParameters(vx_limits=(kmh2ms(-1), kmh2ms(50)),
#                                  acc_limits=(-4, 3),
#                                  delta_max=math.pi / 6,
#                                  ddelta_max=1)
#
#     def __post_init__(self):
#         assert self.vx_limits[0] < self.vx_limits[1]
#         assert self.acc_limits[0] < self.acc_limits[1]
#         assert self.delta_max > 0
#         assert self.ddelta_max > 0
