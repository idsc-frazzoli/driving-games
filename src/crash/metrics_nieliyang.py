from dataclasses import dataclass
from typing import Tuple, Mapping, Dict

import numpy as np

from dg_commons import PlayerName
from sim import CollisionReport
from sim.models import ms2kmh, ModelType, PEDESTRIAN, BICYCLE


@dataclass(frozen=True, unsafe_hash=True)
class NieLiYangRisk:
    p_fatality: float
    """ Probability of fatality """

    def __post_init__(self):
        assert 0 <= self.p_fatality <= 1

    def __str__(self):
        return f"Prob. of fatality: {self.p_fatality * 100:.2f}%\n"


class NieLiYangRiskModel:
    """
    Risk model associated to car-to-pedestrian/cyclist collision
    Source:
    https://www.researchgate.net/publication/260397511_A_Study_of_Fatality_Risk_and_Head_Dynamic_Response_of_Cyclist_and_Pedestrian_Based_on_Passenger_Car_Accident_Data_Analysis_and_Simulations
    Sets the coefficients for the simplest Pedestrian model ("Zero"):
    - Only takes into account delta_v [kilometers per hour]
    """
    coeff: Mapping[ModelType, Tuple] = {  # (a0,a1) = (Intercept, DeltaV)
        PEDESTRIAN: (6.576, -0.092),
        BICYCLE: (6.929, -0.095)
    }

    def _compute_probability(self, v_impact: float, field: ModelType):
        coeff = self.coeff[field]
        weight = coeff[0] + coeff[1] * v_impact
        return 1 / (1 + np.exp(weight))

    def compute_risk(self, v_impact: float, model_type: ModelType) -> NieLiYangRisk:
        return NieLiYangRisk(p_fatality=self._compute_probability(v_impact, model_type))


def compute_NieLiYang_risk(report: CollisionReport, model_types: Mapping[PlayerName, ModelType]) -> \
        Mapping[PlayerName, NieLiYangRisk]:
    """
    Calculates the probability of casualty for the simplest Pedestrian model
    for each player in vehicle-pedestrian crashes, according to the "Pedestrian Zero" model
    :return:
    """
    assert model_types.keys() == report.players.keys()
    assert len(model_types) == 2
    # Variable holding the values of the MalliarisOne coefficients for each severity case
    risk_model = NieLiYangRiskModel()
    damage_reports: Dict[PlayerName, NieLiYangRisk] = {}

    for p, p_collreport in report.players.items():
        if model_types[p] in [PEDESTRIAN, BICYCLE]:
            p_car = set(report.players.keys()).difference(p).pop()
            v_impact = ms2kmh(np.linalg.norm(report.players[p_car].velocity[0]))
            damage_reports[p] = risk_model.compute_risk(v_impact, model_types[p])
        else:
            # Assumption: the probability of fatality for the vehicle when
            # colliding with a pedestrian or cyclist are zero
            damage_reports[p] = NieLiYangRisk(p_fatality=0)

    return damage_reports
