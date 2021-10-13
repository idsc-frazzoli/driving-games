from dataclasses import dataclass
from enum import unique, IntEnum
from math import pi
from typing import Tuple, Mapping, Dict

import numpy as np
from numpy import deg2rad
from shapely.geometry import Point

from dg_commons import X, PlayerName
from dg_commons.sim import CollisionReport
from dg_commons.sim.collision_utils import get_impact_point_direction
from dg_commons.sim.models import ms2mph


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
        return (
            f"Prob. of fatality: {self.p_fatality * 100:.2f}%\n"
            f"Prob. of MAIS 3+ injury: {self.p_mais3 * 100:.2f}%\n"
            f"Prob. of MAIS 2+ injury: {self.p_mais2 * 100:.2f}%\n"
        )


class MalliarisZeroRiskModel:
    """
    Sets the coefficients for the simplest Malliaris model ("Zero"):
    - Only takes into account delta_v [miles per hour]
    """

    coeff: Mapping[MalliarisField, Tuple] = {  # (a0,a1) = (Intercept, DeltaV)
        MalliarisField.FATALITY: (-8.252, 0.177),
        MalliarisField.MAIS3: (-5.450, 0.178),
        MalliarisField.MAIS2: (-3.761, 0.136),
    }

    def _compute_probability(self, delta_v: float, field: MalliarisField):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * delta_v
        return 1 / (1 + np.exp(-weight))

    def compute_risk(self, delta_v: float) -> MalliarisRisk:
        return MalliarisRisk(
            p_fatality=self._compute_probability(delta_v, MalliarisField.FATALITY),
            p_mais3=self._compute_probability(delta_v, MalliarisField.MAIS3),
            p_mais2=self._compute_probability(delta_v, MalliarisField.MAIS2),
        )


class MalliarisOneRiskModel:
    """
    Sets the coefficients for a more complex Malliaris model ("One"):
    - Takes into account delta_v [miles per hour]
    - And the directions of impact dofs, dofb
    """

    coeff: Mapping[MalliarisField, Tuple] = {  # (a0,a1,a2,a3) = (Intercept, DeltaV, DOFS, DOFB)
        MalliarisField.FATALITY: (-9.032, 0.198, 1.462, -1.921),
        MalliarisField.MAIS3: (-5.820, 0.202, 0.556, -2.170),
        MalliarisField.MAIS2: (-4.029, 0.155, 0.465, -1.177),
    }

    def _compute_probability(self, delta_v: float, dofs: int, dofb: int, field: MalliarisField):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * delta_v + coeff[2] * dofs + coeff[3] * dofb
        return 1 / (1 + np.exp(-weight))

    def compute_risk(self, delta_v: float, dofs: int, dofb: int) -> MalliarisRisk:
        return MalliarisRisk(
            p_fatality=self._compute_probability(delta_v, dofs, dofb, MalliarisField.FATALITY),
            p_mais3=self._compute_probability(delta_v, dofs, dofb, MalliarisField.MAIS3),
            p_mais2=self._compute_probability(delta_v, dofs, dofb, MalliarisField.MAIS2),
        )


def compute_malliaris_zero(
    report: CollisionReport, states: Mapping[PlayerName, X]
) -> Mapping[PlayerName, MalliarisRisk]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris Zero" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """
    assert states.keys() == report.players.keys()

    # Variable holding the values of the MalliarisOne coefficients for each severity case
    risk_model = MalliarisZeroRiskModel()
    damage_reports: Dict[PlayerName, MalliarisRisk] = {}

    for key, value in report.players.items():
        delta_v = _get_delta_v(value.velocity[0], value.velocity_after[0])
        damage_reports[key] = risk_model.compute_risk(delta_v)

    return damage_reports


def compute_malliaris_one(
    report: CollisionReport, states: Mapping[PlayerName, X]
) -> Mapping[PlayerName, MalliarisRisk]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris One" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """

    assert states.keys() == report.players.keys()

    # Variable holding the values of the MalliarisOne coefficients for each severity case
    risk_model = MalliarisOneRiskModel()
    damage_reports: Dict[PlayerName, MalliarisRisk] = {}

    for key, value in report.players.items():
        delta_v = _get_delta_v(value.velocity[0], value.velocity_after[0])
        dof = get_malliaris_dof(states[key], report.impact_point)
        damage_reports[key] = risk_model.compute_risk(delta_v, dof[0], dof[1])

    return damage_reports


def _get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init|| in MILES PER HOUR (mph) !!!
    :param v_init:
    :param v_after:
    :return:
    """
    delta_v = v_after - v_init
    return ms2mph(np.linalg.norm(delta_v))


def get_malliaris_dof(state: X, impact_point: Point) -> Tuple[int, int]:
    """
    Get direction of force (DOF) from Malliaris, based on impact_normal
               90º
               |
           ---------
    180º---|       |---> x 0º   (Showcasing angle calculation)
           ---------
               |
              270º
    :param impact_point:
    :param state:
    :return:
    """
    angle_dof = get_impact_point_direction(state=state, impact_point=impact_point)

    if angle_dof < 0:
        angle_dof += 2 * pi
    # Output 0 or 1 based on definitions from Malliaris
    if deg2rad(45) <= angle_dof <= deg2rad(135) or deg2rad(225) <= angle_dof <= deg2rad(315):
        return 1, 0
    elif deg2rad(135) < angle_dof < deg2rad(225):
        return 0, 1
    else:  # 0 < angle < 45 or 315 < angle < 360
        return 0, 0
