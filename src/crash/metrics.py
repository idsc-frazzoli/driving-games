from typing import List, Tuple

import numpy as np
from geometry import SE2_from_xytheta
from shapely.geometry import Point

from crash.metrics_utils import MalliarisZero, MalliarisOne
from sim import CollisionReport, CollisionReportPlayer
from sim.models.vehicle import VehicleState


def get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init||
    """
    delta_v_vec = v_after - v_init
    return delta_v_vec / np.linalg.norm(delta_v_vec)


def get_malliaris_dof(impact_normal: np.array) -> Tuple[int,int]:
    """
    Get direction of force (DOF) from Malliaris, based on impact_normal
    """
    angle = np.arctan2(impact_normal[1], impact_normal[0]) * 180/np.pi

    if angle < 0:
        angle += 360

    if 0 <= angle <= 45 or 135 <= angle <= 225 or 315 <= angle <= 360:
        return 1, 0
    elif 225 < angle < 315:
        return 0, 1
    else:
        return 0, 0

def malliaris_zero(report: CollisionReportPlayer) -> List[float]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """
    tmp = MalliarisZero.coeff_fatality()
    delta_v = get_delta_v(report.velocity[0], report.velocity_after[0])
    p_fatality = tmp.compute_probability(delta_v)

    tmp.coeff_mais3()
    p_mais3 = tmp.compute_probability(delta_v)

    tmp.coeff_mais2()
    p_mais2 = tmp.compute_probability(delta_v)

    return [p_fatality, p_mais3, p_mais2]


def malliaris_one(report_player: CollisionReportPlayer, report: CollisionReport) -> List[float]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
    for each player
    :returns: A list with [p_fatality, p_mais3, p_mais2]
    """
    tmp = MalliarisOne.coeff_fatality()
    delta_v = get_delta_v(report_player.velocity[0], report_player.velocity_after[0])
    dof = get_malliaris_dof(report.impact_normal)

    p_fatality = tmp.compute_probability(delta_v, dof[0], dof[1])

    tmp.coeff_mais3()
    p_mais3 = tmp.compute_probability(delta_v, dof[0], dof[1])

    tmp.coeff_mais2()
    p_mais2 = tmp.compute_probability(delta_v, dof[0], dof[1])

    return [p_fatality, p_mais3, p_mais2]