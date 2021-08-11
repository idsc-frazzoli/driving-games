from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, unsafe_hash=True)
class MalliarisCoeff:

    @classmethod
    @abstractmethod
    def coeff_fatality(cls):
        """
        Initialize coefficients for the computation of casualties
        """
        pass

    @classmethod
    @abstractmethod
    def coeff_mais3(cls):
        """
        Initialize coefficients for the computation of MAIS 3+ injuries
        """
        pass

    @classmethod
    @abstractmethod
    def coeff_mais2(cls):
        """
        Initialize coefficients for the computation of MAIS 2+ injuries
        """
        pass


@dataclass(frozen=True, unsafe_hash=True)
class MalliarisZero(MalliarisCoeff):
    """
    Sets the coefficients for the simplest Malliaris model ("Zero"):
    - Only takes into account delta_v
    """
    a0: float   # Intercept
    a1: float   # DeltaV

    @classmethod
    def coeff_fatality(cls) -> "MalliarisZero":
        MalliarisZero(a0=-8.252, a1=0.177)

    @classmethod
    def coeff_mais3(cls) -> "MalliarisZero":
        MalliarisZero(a0=-5.450, a1=0.178)

    @classmethod
    def coeff_mais2(cls) -> "MalliarisZero":
        MalliarisZero(a0=-3.761, a1=0.136)

    def compute_weight(self, delta_v: float):
        return self.a0 + self.a1 * delta_v

    def compute_probability(self, delta_v: float):
        return 1 / (1 + np.exp(-self.compute_weight(delta_v)))




@dataclass(frozen=True, unsafe_hash=True)
class MalliarisOne(MalliarisCoeff):
    """
    Sets the coefficients for a more complex Malliaris model ("One"):
    - Takes into account delta_v
    - And the directions of impact dofs, dofb
    """
    a0: float   # Intercept
    a1: float   # DeltaV
    a2: float   # DOFS
    a3: float   # DOFB

    @classmethod
    def coeff_fatality(cls) -> "MalliarisOne":
        MalliarisOne(a0=-9.032, a1=0.198, a2=1.462, a3=-1.921)

    @classmethod
    def coeff_mais3(cls) -> "MalliarisOne":
        MalliarisOne(a0=-5.820, a1=0.202, a2=0.556, a3=-2.170)

    @classmethod
    def coeff_mais2(cls) -> "MalliarisOne":
        MalliarisOne(a0=-4.029, a1=0.155, a2=0.465, a3=-1.177)

    def compute_weight(self, delta_v: float, dofs: bool, dofb: bool):
        return self.a0 + self.a1 * delta_v + self.a2 * dofs + self.a3 * dofb

    def compute_probability(self, delta_v: float, dofs: bool, dofb: bool):
        return 1 / (1 + np.exp(-self.compute_weight(delta_v, dofs, dofb)))
