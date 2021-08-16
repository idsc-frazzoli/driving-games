from typing import List, Tuple

import numpy as np
from geometry import SE2_from_xytheta, SE2value
from shapely.geometry import Point, Polygon

from crash.metrics_structures import MalliarisOneReportPlayer, MetricsReport
from crash.metrics_utils import MalliarisZero, MalliarisOne
from games import X, PlayerName
from sim import CollisionReport, CollisionReportPlayer
from sim.models.vehicle import VehicleState


def get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init|| in MILES PER HOUR (mph) !!!
    :param v_init:
    :param v_after:
    :return:
    """
    delta_v = v_after - v_init
    return np.linalg.norm(delta_v) * 2.23694


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
    else: # 0 < angle < 45 or 315 < angle < 360
        return 0, 0


def malliaris_zero(a: PlayerName, b: PlayerName, report: CollisionReport) -> MetricsReport:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris Zero" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """

    p_fatality = []
    p_mais3 = []
    p_mais2 = []

    # Variable holding the values of the MalliarisZero coefficients for each severity case
    tmp = MalliarisZero()

    for key, value in report.players.items():

        # Compute probability of fatality
        tmp.coeff_fatality()
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        p_fatality.append(tmp.compute_probability(delta_v))

        # Compute probability of MAIS 3+ injury
        tmp.coeff_mais3()
        p_mais3.append(tmp.compute_probability(delta_v))

        # Compute probability of MAIS 2+ injury
        tmp.coeff_mais2()
        p_mais2.append(tmp.compute_probability(delta_v))

    a_metrics = MalliarisOneReportPlayer(p_fatality=p_fatality[0],
                                         p_mais3=p_mais3[0],
                                         p_mais2=p_mais2[0])

    b_metrics = MalliarisOneReportPlayer(p_fatality=p_fatality[1],
                                         p_mais3=p_mais3[1],
                                         p_mais2=p_mais2[1])

    return MetricsReport(players={a: a_metrics, b: b_metrics})


def malliaris_one(a: PlayerName, b: PlayerName, report: CollisionReport, a_state: X, b_state: X) -> MetricsReport:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris One" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """

    p_fatality = []
    p_mais3 = []
    p_mais2 = []

    # Variable holding the values of the MalliarisOne coefficients for each severity case
    tmp = MalliarisOne()

    states = [a_state, b_state]

    count = 0
    for key, value in report.players.items():

        # Compute probability of fatality
        tmp.coeff_fatality()
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        dof = get_malliaris_dof(report.impact_point, value.velocity[0], value.footprint, states[count])
        p_fatality.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

        # Compute probability of MAIS 3+ injury
        tmp.coeff_mais3()
        p_mais3.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

        # Compute probability of MAIS 2+ injury
        tmp.coeff_mais2()
        p_mais2.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

        count += 1

    a_metrics = MalliarisOneReportPlayer(p_fatality=p_fatality[0],
                                         p_mais3=p_mais3[0],
                                         p_mais2=p_mais2[0])

    b_metrics = MalliarisOneReportPlayer(p_fatality=p_fatality[1],
                                         p_mais3=p_mais3[1],
                                         p_mais2=p_mais2[1])

    return MetricsReport(players={a: a_metrics, b: b_metrics})
