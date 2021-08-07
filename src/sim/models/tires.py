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
        return Pacejka(B=10, C=1.9, D=1)

    @classmethod
    def default_car_rear(cls):
        return Pacejka(B=11, C=1.9, D=1)

    @classmethod
    def default_bicycle_front(cls):
        logger.warn("pacejka for bikes is not tuned")
        # todo
        return Pacejka(B=12, C=2.3, D=0.8)

    @classmethod
    def default_bicycle_rear(cls):
        logger.warn("pacejka for bikes is not tuned")
        # todo
        return Pacejka(B=5, C=2, D=0.3)


@dataclass(frozen=True)
class Pacejka4p(Pacejka):
    E: float = 0.9
    "curvature"

    def evaluate(self, slip: float) -> float:
        return self.D * sin(self.C * atan(self.B * slip - self.E * (self.B * slip - self.E * atan(self.B * slip))))
