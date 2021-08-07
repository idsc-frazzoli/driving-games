from dataclasses import dataclass
from math import atan, sin

__all__ = ["Pacejka", "Pacejka4p"]

from sim import logger


@dataclass(frozen=True)
class Pacejka:
    B: float
    "stiffness"
    C: float
    "shape"
    D: float  # gravity acceleration considered
    "peak"

    def evaluate(self, slip: float) -> float:
        return self.D * sin(self.C * atan(self.B * slip))

    @classmethod
    def default_car_front(cls):
        return Pacejka(B=11, C=1.3, D=4000)

    @classmethod
    def default_car_rear(cls):
        return Pacejka(B=12.7, C=1.3, D=3800)

    @classmethod
    def default_bicycle_front(cls):
        logger.warn("pacejka for bikes is not tuned")
        # todo
        return Pacejka(B=11, C=1.3, D=100)

    @classmethod
    def default_bicycle_rear(cls):
        logger.warn("pacejka for bikes is not tuned")
        # todo
        return Pacejka(B=12.7, C=1.3, D=100)


@dataclass(frozen=True)
class Pacejka4p(Pacejka):
    E: float
    "curvature"

    def evaluate(self, slip: float) -> float:
        return self.D * sin(self.C * atan(self.B * slip - self.E * (self.B * slip - self.E * atan(self.B * slip))))
