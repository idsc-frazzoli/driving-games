from typing import List

import numpy as np
from crash.metrics_utils import MalliarisZero
from sim import CollisionReport, CollisionReportPlayer


def get_delta_v(v_init: np.ndarray, v_after: np.ndarray) -> float:
    """
    Computes the norm of delta_v -> ||v_after - v_init||
    """
    delta_v_vec = v_after - v_init
    return delta_v_vec / np.linalg.norm(delta_v_vec)


def malliaris_zero(report: CollisionReportPlayer) -> List[float]:
    """
    Calculates the probability of casualty, MAIS 3+ and MAIS 2+ for the simplest Malliaris model
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