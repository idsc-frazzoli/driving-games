from abc import abstractmethod

import numpy as np


class MalliarisCoeff:

    @abstractmethod
    def coeff_fatality(self):
        """
        Initialize coefficients for the computation of casualties
        """
        pass

    @abstractmethod
    def coeff_mais3(self):
        """
        Initialize coefficients for the computation of MAIS 3+ injuries
        """
        pass

    @abstractmethod
    def coeff_mais2(self):
        """
        Initialize coefficients for the computation of MAIS 2+ injuries
        """
        pass


class MalliarisZero(MalliarisCoeff):
    """
    Sets the coefficients for the simplest Malliaris model ("Zero"):
    - Only takes into account delta_v [miles per hour]
    """

    # a0: float   # Intercept
    # a1: float   # DeltaV

    def __init__(self):
        self.a0 = -8.252
        self.a1 = 0.177

    def coeff_fatality(self):
        self.a0 = -8.252
        self.a1 = 0.177

    def coeff_mais3(self):
        self.a0 = -5.450
        self.a1 = 0.178

    def coeff_mais2(self):
        self.a0 = -3.761
        self.a1 = 0.136

    def compute_weight(self, delta_v: float):
        return self.a0 + self.a1 * delta_v

    def compute_probability(self, delta_v: float):
        return 1 / (1 + np.exp(-self.compute_weight(delta_v)))


class MalliarisOne(MalliarisCoeff):
    """
    Sets the coefficients for a more complex Malliaris model ("One"):
    - Takes into account delta_v [miles per hour]
    - And the directions of impact dofs, dofb
    """
    a0: float  # Intercept
    a1: float  # DeltaV
    a2: float  # DOFS
    a3: float  # DOFB

    def __init__(self):
        self.a0 = -9.032
        self.a1 = 0.198
        self.a2 = 1.462
        self.a3 = -1.921

    def coeff_fatality(self):
        self.a0 = -9.032
        self.a1 = 0.198
        self.a2 = 1.462
        self.a3 = -1.921

    def coeff_mais3(self):
        self.a0 = -5.820
        self.a1 = 0.202
        self.a2 = 0.556
        self.a3 = -2.170

    def coeff_mais2(self):
        self.a0 = -4.029
        self.a1 = 0.155
        self.a2 = 0.465
        self.a3 = -1.177

    def compute_weight(self, delta_v: float, dofs: int, dofb: int):
        return self.a0 + self.a1 * delta_v + self.a2 * dofs + self.a3 * dofb

    def compute_probability(self, delta_v: float, dofs: int, dofb: int):
        return 1 / (1 + np.exp(-self.compute_weight(delta_v, dofs, dofb)))
