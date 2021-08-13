from typing import List, Tuple

import numpy as np
from geometry import SE2_from_xytheta
from shapely.geometry import Point, Polygon

from crash.metrics_utils import MalliarisZero, MalliarisOne
from sim import CollisionReport, CollisionReportPlayer
from sim.models.vehicle import VehicleState


def get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init|| in MILES PER HOUR
    :param v_init:
    :param v_after:
    :return:
    """
    delta_v = v_after - v_init
    return np.linalg.norm(delta_v) * 2.23694


def get_malliaris_dof(impact_point: Point, footprint: Polygon) -> Tuple[int, int]:
    """
    Get direction of force (DOF) from Malliaris, based on impact_normal
    :param impact_point:
    :param footprint:
    :return:
    """

    dof = [impact_point.x - footprint.centroid.x, impact_point.y - footprint.centroid.y]  # Direction of Force (DOF) coordinates
    angle = np.arctan2(dof[1], dof[0]) * 180 / np.pi  # DOF angle based on coordinates

    if angle < 0:
        angle += 360

    # Output 0 or 1 based on definitions from Malliaris
    if 0 <= angle <= 45 or 135 <= angle <= 225 or 315 <= angle <= 360:
        return 1, 0
    elif 225 < angle < 315:
        return 0, 1
    else:
        return 0, 0


def malliaris_zero(report: CollisionReport) -> List[List[float]]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris Zero" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """

    p_fatality = []
    p_mais3 = []
    p_mais2 = []

    for key, value in report.players.items():

        tmp = MalliarisZero.coeff_fatality()
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        p_fatality.append(tmp.compute_probability(delta_v))

        tmp = tmp.coeff_mais3()
        p_mais3.append(tmp.compute_probability(delta_v))

        tmp = tmp.coeff_mais2()
        p_mais2.append(tmp.compute_probability(delta_v))

    return [p_fatality, p_mais3, p_mais2]


def malliaris_one(report: CollisionReport) -> List[List[float]]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player in two vehicles crashes, according to the "Malliaris One" model
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """

    p_fatality = []
    p_mais3 = []
    p_mais2 = []

    for key, value in report.players.items():
        tmp = MalliarisOne.coeff_fatality()
        delta_v = get_delta_v(value.velocity[0], value.velocity_after[0])
        #dof = get_malliaris_dof_old(report.impact_normal)
        dof = get_malliaris_dof(report.impact_point, value.footprint)
        p_fatality.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

        tmp = tmp.coeff_mais3()
        p_mais3.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

        tmp = tmp.coeff_mais2()
        p_mais2.append(tmp.compute_probability(delta_v, dof[0], dof[1]))

    return [p_fatality, p_mais3, p_mais2]
