from dataclasses import dataclass
from enum import unique, IntEnum
from typing import Tuple, Mapping

import numpy as np


@unique
class MalliarisField(IntEnum):
    FATALITY = 0
    MAIS3 = 1
    MAIS2 = 2


@dataclass(frozen=True, unsafe_hash=True)
class MalliarisRisk:
    """
    # todo put reference to paper
    """
    p_fatality: float
    """ Probability of fatality """
    p_mais3: float
    """ Probability of MAIS 3+ injury """
    p_mais2: float
    """ Probability of MAIS 2+ injury """

    def __post_init__(self):
        assert 0 <= self.p_fatality <= 1
        assert 0 <= self.p_mais3 <= 1
        assert 0 <= self.p_mais2 <= 1

    def __str__(self):
        return f"Prob. of fatality: {self.p_fatality * 100:.2f}%\n" \
               f"Prob. of MAIS 3+ injury: {self.p_mais3 * 100:.2f}%\n" \
               f"Prob. of MAIS 2+ injury: {self.p_mais2 * 100:.2f}%\n"


class MalliarisZeroRiskModel:
    """
    Sets the coefficients for the simplest Malliaris model ("Zero"):
    - Only takes into account delta_v [miles per hour]
    """
    coeff: Mapping[MalliarisField, Tuple] = {  # (a0,a1) = (Intercept, DeltaV)
        MalliarisField.FATALITY: (-8.252, 0.177),
        MalliarisField.MAIS3: (-5.450, 0.178),
        MalliarisField.MAIS2: (-3.761, 0.136)}

    def _compute_probability(self, delta_v: float, field: MalliarisField):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * delta_v
        return 1 / (1 + np.exp(-weight))

    def compute_risk(self, delta_v: float) -> MalliarisRisk:
        return MalliarisRisk(p_fatality=self._compute_probability(delta_v, MalliarisField.FATALITY),
                             p_mais3=self._compute_probability(delta_v, MalliarisField.MAIS3),
                             p_mais2=self._compute_probability(delta_v, MalliarisField.MAIS2))


class MalliarisOneRiskModel:
    """
    Sets the coefficients for a more complex Malliaris model ("One"):
    - Takes into account delta_v [miles per hour]
    - And the directions of impact dofs, dofb
    """

    coeff: Mapping[MalliarisField, Tuple] = {  # (a0,a1,a2,a3) = (Intercept, DeltaV, DOFS, DOFB)
        MalliarisField.FATALITY: (-9.032, 0.198, 1.462, -1.921),
        MalliarisField.MAIS3: (-5.820, 0.202, 0.556, -2.170),
        MalliarisField.MAIS2: (-4.029, 0.155, 0.465, -1.177)}

    def _compute_probability(self, delta_v: float, dofs: int, dofb: int, field: MalliarisField):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * delta_v + coeff[2] * dofs + coeff[3] * dofb
        return 1 / (1 + np.exp(-weight))

    def compute_risk(self, delta_v: float, dofs: int, dofb: int) -> MalliarisRisk:
        return MalliarisRisk(p_fatality=self._compute_probability(delta_v, dofs, dofb, MalliarisField.FATALITY),
                             p_mais3=self._compute_probability(delta_v, dofs, dofb, MalliarisField.MAIS3),
                             p_mais2=self._compute_probability(delta_v, dofs, dofb, MalliarisField.MAIS2))
