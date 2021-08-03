from typing import Mapping, List

import numpy as np
from commonroad_dc import pycrcc
from geometry import SE2_from_xytheta, SE2value

from sim import ImpactLocation, IMPACT_FRONT, IMPACT_BACK, IMPACT_LEFT, IMPACT_RIGHT
from sim.models.vehicle import VehicleState


def get_vertices_from_rectangle(rect: pycrcc.RectOBB) -> List[List[float]]:
    """ This gets the car vertices in the global reference frame (RF)
    1---------0
    |         |-----> x
    2---------3
    :param rect:
    :return:
    """

    l2g: SE2value = SE2_from_xytheta((rect.center()[0], rect.center()[1], rect.orientation()))
    # vertices of the rectangle
    vertices = np.array([[rect.r_y(), -rect.r_y(), -rect.r_y(), rect.r_y()],
                         [rect.r_x(), rect.r_x(), -rect.r_x(), -rect.r_x()],
                         [1, 1, 1, 1]])

    vertices = l2g @ vertices
    vertices = vertices[:-1, :]  # Remove last row
    return vertices.T.tolist()


def get_rectangle_mesh(footprint: pycrcc.RectOBB) -> Mapping[ImpactLocation, pycrcc.Triangle]:
    """
    This returns all the vertices of a rectangle in the global reference frame of the map (a bit useless for now)
    :return:
    """

    vertices = get_vertices_from_rectangle(footprint)
    rect_cx, rect_cy = footprint.center()

    impact_locations: Mapping[ImpactLocation, pycrcc.Triangle] = {
        IMPACT_FRONT: pycrcc.Triangle(*vertices[0], *vertices[3], *(rect_cx, rect_cy)),
        IMPACT_BACK: pycrcc.Triangle(*vertices[1], *vertices[2], *(rect_cx, rect_cy)),
        IMPACT_LEFT: pycrcc.Triangle(*vertices[0], *vertices[1], *(rect_cx, rect_cy)),
        IMPACT_RIGHT: pycrcc.Triangle(*vertices[2], *vertices[3], *(rect_cx, rect_cy)),
    }
    return impact_locations
