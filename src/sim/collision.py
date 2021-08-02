from dataclasses import dataclass
from typing import NewType, Mapping, List

import numpy as np
from commonroad_dc import pycrcc
from geometry import SE2_from_xytheta, SE2value

ImpactLocation = NewType("ImpactLocation", str)
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_LEFT = ImpactLocation("left")
IMPACT_RIGHT = ImpactLocation('right')

# todo for now just a bool in the future we want more detailed info
CollisionBool = bool


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReport:
    location: List[ImpactLocation]
    """ Location of the impact """
    at_fault: bool
    """ AT fault is defined as...."""
    rel_velocity: float
    """ Relative velocity defined as... """
    energy_transfer: float
    """ """


def get_vertices_g(rect: pycrcc.RectOBB) -> List[List[float]]:
    """This gets the car vertices in the global reference frame (RF)
    1---------0
    |         |-----> x
    2---------3
    :param rect:
    :return:"""

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

    vertices = get_vertices_g(footprint)
    rect_cx, rect_cy = footprint.center()

    impactlocations: Mapping[ImpactLocation, pycrcc.Triangle] = {
        IMPACT_FRONT: pycrcc.Triangle(*vertices[0], *vertices[3], *(rect_cx, rect_cy)),
        IMPACT_BACK: pycrcc.Triangle(*vertices[1], *vertices[2], *(rect_cx, rect_cy)),
        IMPACT_LEFT: pycrcc.Triangle(*vertices[0], *vertices[1], *(rect_cx, rect_cy)),
        IMPACT_RIGHT: pycrcc.Triangle(*vertices[2], *vertices[3], *(rect_cx, rect_cy)),
    }
    return impactlocations


def is_a_at_fault(a: pycrcc.RectOBB, b: pycrcc.RectOBB):
    # this will need to be implemented with some better logic
    return False


def compute_collision_report(a: pycrcc.RectOBB, b: pycrcc.RectOBB) -> CollisionReport:
    """
    Compute the collision report for A given that B collided
    """

    a_mesh = get_rectangle_mesh(a)
    locations: List[ImpactLocation] = []
    for loc, zone in a_mesh.items():
        if b.collide(zone):
            locations.append(loc)
    if not locations:
        raise RuntimeWarning("Detected a collision but unable to find the impact location")
    at_fault: bool = is_a_at_fault(a, b)
    # todo compute all the rest
    return CollisionReport(location=locations,
                           at_fault=at_fault,
                           rel_velocity=0,
                           energy_transfer=0
                           )
