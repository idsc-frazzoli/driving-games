from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Sequence

from games import PlayerName
from sim import Color

Colour = Tuple[float, float, float]
""" An alias to store the RGB values of a colour """

black = PlayerName("black")
forestgreen = PlayerName("forestgreen")
firebrick = PlayerName("firebrick")
dodgerblue = PlayerName("dodgerblue")
orchid = PlayerName("orchid")


@dataclass(frozen=True, unsafe_hash=True)
class ModelGeometry(ABC):
    m: float
    """ Vehicle Mass [kg] """
    Iz: float
    """ Moment of inertia (used only in the dynamic model) """
    e: float
    """ Restitution coefficient (used only in collisions energy transfer). 
    Ratio of the differences in vehicle speeds before and after the collision -> 0 < e < 1"""
    color: Color
    """ Color must be able to be parsed by matplotlib"""

    @property
    @abstractmethod
    def outline(self) -> Sequence[Tuple[float, float]]:
        pass


@dataclass(frozen=True, unsafe_hash=True)
class ModelParameters:
    vx_limits: Tuple[float, float]
    """ Minimum and Maximum velocities [m/s] """
    acc_limits: Tuple[float, float]
    """ Minimum and Maximum acceleration [m/s^2] """

    def __post_init__(self):
        assert self.vx_limits[0] < self.vx_limits[1]
        assert self.acc_limits[0] < self.acc_limits[1]
