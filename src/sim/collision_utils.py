from typing import Mapping, List, Tuple, Any

import numpy as np
from commonroad_dc import pycrcc
from geometry import SE2_from_xytheta, SE2value
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from sim import ImpactLocation, IMPACT_FRONT, IMPACT_BACK, IMPACT_LEFT, IMPACT_RIGHT
from sim.models.vehicle import VehicleState


def get_vertices_as_list(rect: pycrcc.RectOBB) -> List[List[float]]:
    """ This gets the car vertices in the global reference frame (RF) as a list
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


def get_vertices_as_tuple(rect: pycrcc.RectOBB) -> Tuple[Tuple[float, ...], ...]:
    """ This gets the car vertices in the global reference frame (RF) as a tuple
    1---------0
    |         |-----> x
    2---------3
    :param rect:
    :return:
    """
    vertices = get_vertices_as_list(rect)
    return tuple([tuple(row) for row in vertices])


def get_rectangle_mesh(footprint: pycrcc.RectOBB) -> Mapping[ImpactLocation, pycrcc.Triangle]:
    """
    This returns all the vertices of a rectangle in the global reference frame of the map (a bit useless for now)
    :return:
    """

    vertices = get_vertices_as_list(footprint)
    rect_cx, rect_cy = footprint.center()

    impact_locations: Mapping[ImpactLocation, pycrcc.Triangle] = {
        IMPACT_FRONT: pycrcc.Triangle(*vertices[0], *vertices[3], *(rect_cx, rect_cy)),
        IMPACT_BACK: pycrcc.Triangle(*vertices[1], *vertices[2], *(rect_cx, rect_cy)),
        IMPACT_LEFT: pycrcc.Triangle(*vertices[0], *vertices[1], *(rect_cx, rect_cy)),
        IMPACT_RIGHT: pycrcc.Triangle(*vertices[2], *vertices[3], *(rect_cx, rect_cy)),
    }
    return impact_locations


def get_nearest_collision_points(vehicle_a: pycrcc.RectOBB, vehicle_b: pycrcc.RectOBB) -> np.ndarray:
    center = Point(vehicle_a.center()[0], vehicle_a.center()[1])
    vertices_b = get_vertices_as_tuple(vehicle_b)
    vertices_b += (vertices_b[0],)
    rect = Polygon(vertices_b)
    nearest_pts = nearest_points(center, rect)
    return np.array([[nearest_pts[0].x, nearest_pts[0].y], [nearest_pts[1].x, nearest_pts[1].y]])


def get_normal_of_impact(vehicle_a: pycrcc.RectOBB, vehicle_b: pycrcc.RectOBB) -> np.ndarray:
    nearest_pts = get_nearest_collision_points(vehicle_a, vehicle_b)
    n = nearest_pts[1] - nearest_pts[0]  # Subtract nearest_point_b - center_of_a
    n /= np.linalg.norm(n)               # Make it a unitary vector
    return n

