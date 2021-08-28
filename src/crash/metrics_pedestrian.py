from dataclasses import dataclass
from enum import unique, IntEnum
from typing import Tuple, Mapping, Dict

import numpy as np

from games import PlayerName, X
from sim import CollisionReport
from sim.models import ms2kmh
from sim.models.pedestrian import PedestrianState


@unique
class PedestrianField(IntEnum):
    FATALITY = 0


@dataclass(frozen=True, unsafe_hash=True)
class PedestrianRisk:
    """
    # todo put reference to paper
    """
    p_fatality: float
    """ Probability of fatality """

    def __post_init__(self):
        assert 0 <= self.p_fatality <= 1

    def __str__(self):
        return f"Prob. of fatality: {self.p_fatality * 100:.2f}%\n"


class PedestrianZeroRiskModel:
    """
    Sets the coefficients for the simplest Pedestrian model ("Zero"):
    - Only takes into account delta_v [kilometers per hour]
    """
    coeff: Mapping[PedestrianField, Tuple] = {  # (a0,a1) = (Intercept, DeltaV)
        PedestrianField.FATALITY: (6.576, -0.092)}

    def _compute_probability(self, delta_v: float, field: PedestrianField):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * delta_v
        return 1 / (1 + np.exp(weight))

    def compute_risk(self, delta_v: float) -> PedestrianRisk:
        return PedestrianRisk(p_fatality=self._compute_probability(delta_v, PedestrianField.FATALITY))


def compute_pedestrian_zero(report: CollisionReport, states: Mapping[PlayerName, X]) -> \
        Mapping[PlayerName, PedestrianRisk]:
    """
    Calculates the probability of casualty for the simplest Pedestrian model
    for each player in vehicle-pedestrian crashes, according to the "Pedestrian Zero" model
    :return:
    """
    assert states.keys() == report.players.keys()

    # Variable holding the values of the MalliarisOne coefficients for each severity case
    risk_model = PedestrianZeroRiskModel()
    damage_reports: Dict[PlayerName, PedestrianRisk] = {}

    for key, value in report.players.items():
        delta_v = _get_delta_v(value.velocity[0], value.velocity_after[0])
        if isinstance(states[key], PedestrianState):
            damage_reports[key] = risk_model.compute_risk(delta_v)
        else:
            # Assumption: the probability of casualty for the vehicle when
            # colliding with a pedestrian are zero
            damage_reports[key] = PedestrianRisk(p_fatality=0)

    return damage_reports


def _get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init|| in MILES PER HOUR (mph) !!!
    :param v_init:
    :param v_after:
    :return:
    """
    delta_v = v_after - v_init
    return ms2kmh(np.linalg.norm(delta_v))