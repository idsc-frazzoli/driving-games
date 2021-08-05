from typing import Mapping, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from sim import ImpactLocation, IMPACT_FRONT, IMPACT_BACK, IMPACT_LEFT, IMPACT_RIGHT
from sim.models.vehicle_structures import ModelGeometry


def get_rectangle_mesh(footprint: Polygon) -> Mapping[ImpactLocation, Polygon]:
    """
    This returns all the vertices of a rectangle in the global reference frame of the map (a bit useless for now)
    :param footprint: RectOBB object
    :return:
    """

    vertices = footprint.exterior.coords[:-1]  # todo check the order!
    cxy = footprint.centroid.coords[0]
    # maybe we can use triangulate from shapely
    impact_locations: Mapping[ImpactLocation, Polygon] = {
        IMPACT_FRONT: Polygon([cxy, vertices[0], vertices[3], cxy]),
        IMPACT_BACK: Polygon([cxy, vertices[1], vertices[2], cxy]),
        IMPACT_LEFT: Polygon([cxy, vertices[0], vertices[1], cxy]),
        IMPACT_RIGHT: Polygon([cxy, vertices[2], vertices[3], cxy]),
    }
    for shape in impact_locations.values():
        assert shape.is_valid
    return impact_locations


def _find_intersection_points(a: Polygon, b: Polygon) -> List[Tuple[float, float]]:
    int_shape = a.intersection(b)
    points = list(int_shape.exterior.coords[:-1])
    plt.plot(*a.exterior.xy, "b")
    plt.plot(*b.exterior.xy, "r")
    for p in points:
        plt.plot(*p, "o")

    def is_contained_in_aorb(p) -> bool:
        shapely_point = Point(p).buffer(1.0e-9)
        print("blu:", a.contains(shapely_point))
        print("red:",b.contains(shapely_point))
        return a.contains(shapely_point) or b.contains(shapely_point)

    points[:] = [p for p in points if not is_contained_in_aorb(p)]
    for p in points:
        plt.plot(*p, "x")

    plt.savefig("test.png")
    if not len(points) == 2:
        raise RuntimeError(f"At the moment collisions with {len(points)} intersecting points are not supported")
    return points


def compute_impact_geometry(a: Polygon, b: Polygon) -> (np.ndarray, Point):
    """
    This computes the normal of impact between vehicles a and b
    :param a: RectOBB object
    :param b: RectOBB object
    :return:
    """
    assert not a.touches(b)
    intersecting_points = _find_intersection_points(a, b)

    # todo: important -> fix this making sure that n is the same wrt a and wrt b
    # fixme this approximations works well only with circles
    nearest_pt = nearest_points(a.centroid, b)[1]
    n = np.array(nearest_pt.coords[0]) - np.array(a.centroid.coords[0])  # Subtract nearest_point_b - center_of_a
    n /= np.linalg.norm(n)  # Make it a unitary vector

    return n, Point([0, 0])


def get_tangent_of_impact(n: np.ndarray, rel_v: np.ndarray) -> np.ndarray:
    """
    This computes the tangent of impact between vehicles a and b
    :param n: Normal of impact
    :param rel_v: Relative velocity between a and b
    :return:
    """
    # fixme just take the orthogonal vector to n???
    t = rel_v - np.dot(rel_v, n) * n
    t /= np.linalg.norm(t)
    return t


def compute_impulse_response(vec: np.ndarray,
                             rel_v: np.ndarray,
                             r_ap: np.ndarray,
                             r_bp: np.ndarray,
                             a_geom: ModelGeometry,
                             b_geom: ModelGeometry) -> float:
    """
    This computes the impulse scalar
    :param vec:             Vector onto which to project rel_v (normally, n or t)
    :param rel_v:           Relative velocity between a and b
    :param r_ap:            Vector from CG of a to collision point P
    :param r_bp:            Vector from CG of b to collision point P
    :param a_geom:          Geometry of vehicle a
    :param b_geom:          Geometry of vehicle b
    :return:
    """
    e = min(a_geom.e, b_geom.e)  # Restitution coefficient -> represents the "bounciness" of the vehicle
    rel_v_along_vec = np.dot(rel_v, vec)
    rel_v_along_vec = np.linalg.norm(rel_v_along_vec)
    j = -(1 + e) * rel_v_along_vec
    tmp = (np.dot(r_ap, vec) ** 2 / a_geom.Iz) + (np.dot(r_bp, vec) ** 2 / b_geom.Iz)
    j /= (1 / a_geom.m + 1 / b_geom.m + tmp)
    return j


def get_velocity_after_collision(n: np.ndarray, v_initial: np.ndarray, m: float, j: float) -> np.ndarray:
    """
    This computes the velocity after the collision based on the impulse resolution method
    :param n:           normal of impact
    :param v_initial:   velocity right before the collision
    :param m:           vehicle mass
    :param j:           impulse scalar
    :return:
    """
    # todo rotational!!
    return v_initial + (j * n) / m


def kinetic_energy(velocity: np.ndarray, m: float) -> float:
    """
    This computes the kinetic energy lost in the collision as 1/2*m*(vf^2-vi^2)
    :param velocity:   velocity right before the collision
    :param m:     mass of the object
    :return:
    """
    # todo also rotational components?
    return .5 * m * np.linalg.norm(velocity) ** 2
