from typing import Tuple, Mapping, Dict

import numpy as np
from shapely.geometry import Point, Polygon

from crash.metrics_utils import MalliarisOneRiskModel, MalliarisZeroRiskModel, MalliarisRisk
from games import X, PlayerName
from sim import CollisionReport
from sim.models import ms2mph


def get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init|| in MILES PER HOUR (mph) !!!
    :param v_init:
    :param v_after:
    :return:
    """
    delta_v = v_after - v_init
    return ms2mph(np.linalg.norm(delta_v))


def get_malliaris_dof(impact_point: Point, v_init: np.array, footprint: Polygon, state: X) -> Tuple[int, int]:
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
    :param v_init:
    :param footprint:
    :param state:
    :return:
    """

    # Option 1
    # Car heading unit vector
    car_heading = np.array([np.cos(state.theta), np.sin(state.theta)])
    # Direction of Force (DOF) -> vector that goes from car center to impact point
    dof = np.array([impact_point.x - footprint.centroid.x,
                    impact_point.y - footprint.centroid.y])  # Direction of Force (DOF) coordinates
    # DOF unit vector
    dof /= np.linalg.norm(dof)
    # Angle between DOF and car heading calculation (taking into account the sign)
    angle = np.arctan2(car_heading[0] * dof[1] - car_heading[1] * dof[0],
                       car_heading[0] * dof[0] + car_heading[1] * dof[1]) * 180 / np.pi
    # Option 2
    '''
    impact_point_g = np.array([[impact_point.x], [impact_point.y], [1]])
    g2l: SE2value = SE2_from_xytheta((-footprint.centroid.x, -footprint.centroid.y, -state.theta))
    impact_point_l = g2l @ impact_point_g
    angle = np.arctan2(impact_point_l[1], impact_point_l[0]) * 180 / np.pi
    '''
    if angle < 0:
        angle += 360

    # Output 0 or 1 based on definitions from Malliaris
    if 45 <= angle <= 135 or 225 <= angle <= 315:
        return 1, 0
    elif 135 < angle < 225:
        return 0, 1
    else:  # 0 < angle < 45 or 315 < angle < 360
        return 0, 0


def malliaris_zero(report: CollisionReport, states: Mapping[PlayerName, X]) -> \
        Mapping[PlayerName, MalliarisRisk]:
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
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        damage_reports[key] = risk_model.compute_risk(delta_v)

    return damage_reports


def malliaris_one(report: CollisionReport, states: Mapping[PlayerName, X]) -> \
        Mapping[PlayerName, MalliarisRisk]:
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
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        dof = get_malliaris_dof(report.impact_point, value.velocity[0], value.footprint, states[key])
        damage_reports[key] = risk_model.compute_risk(delta_v, dof[0], dof[1])

    return damage_reports
