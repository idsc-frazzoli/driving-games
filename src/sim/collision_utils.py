from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, Point, LineString

from sim.models.vehicle_structures import VehicleGeometry


def _find_intersection_points(a_shape: Polygon, b_shape: Polygon) -> List[Tuple[float, float]]:
    int_shape = a_shape.intersection(b_shape)
    points = list(int_shape.exterior.coords[:-1])

    def is_contained_in_aorb(p) -> bool:
        shapely_point = Point(p).buffer(1.0e-9)
        return a_shape.contains(shapely_point) or b_shape.contains(shapely_point)

    points[:] = [p for p in points if not is_contained_in_aorb(p)]
    if not len(points) == 2:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(*a_shape.exterior.xy, "b")
        plt.plot(*b_shape.exterior.xy, "r")
        for p in points:
            plt.plot(*p, "o")
        plt.savefig("debug.png")
        raise RuntimeError(f"At the moment collisions with {len(points)} intersecting points are not supported")
    return points


def compute_impact_geometry(a: Polygon, b: Polygon) -> (np.ndarray, Point):
    """
    This computes the normal of impact between vehicles a and b
    :param a: Polygon object
    :param b: Polygon object
    :return:
    """
    assert not a.touches(b)
    intersecting_points = _find_intersection_points(a, b)
    impact_point = LineString(intersecting_points).interpolate(0.5, normalized=True)
    first, second = intersecting_points
    dxdy_surface = (second[0] - first[0], second[1] - first[1])
    normal = np.array([-dxdy_surface[1], dxdy_surface[0]])
    normal /= np.linalg.norm(normal)
    r_ap = np.array(impact_point.coords[0]) - np.array(a.centroid.coords[0])
    if np.dot(r_ap, normal) < 0:
        # rotate by 180 if pointing into the inwards of A
        normal *= -1
    return normal, impact_point


def compute_impulse_response(n: np.ndarray,
                             vel_ab: np.ndarray,
                             r_ap: np.ndarray,
                             r_bp: np.ndarray,
                             a_geom: VehicleGeometry,
                             b_geom: VehicleGeometry) -> float:
    """
    The impulse J is defined in terms of force F and time period ∆t
    J = F*∆t = ma*∆t = m *∆v/∆t *∆t = m*∆v
    :param n:             Vector onto which to project rel_v (normally, n or t)
    :param vel_ab:           Relative velocity between a and b
    :param r_ap:            Vector from CG of a to collision point P
    :param r_bp:            Vector from CG of b to collision point P
    :param a_geom:          Geometry of vehicle a
    :param b_geom:          Geometry of vehicle b
    :return:
    """
    # Restitution coefficient -> represents the "bounciness" of the vehicle
    e = min(a_geom.e, b_geom.e)
    j = -(1 + e) * np.dot(vel_ab, n)
    rot_part = (np.cross(r_ap, n) ** 2 / a_geom.Iz) + (np.cross(r_bp, n) ** 2 / b_geom.Iz)
    j /= (1 / a_geom.m + 1 / b_geom.m + rot_part)
    return j


def velocity_after_collision(n: np.ndarray, velocity: np.ndarray, m: float, j: float) -> np.ndarray:
    """
    This computes the velocity after the collision based on the impulse resolution method
    :param n:           normal of impact
    :param velocity:   velocity right before the collision
    :param m:           vehicle mass
    :param j:           impulse scalar
    :return: velocity after the impulse has been applied
    """
    return velocity + (j * n) / m


def rot_velocity_after_collision(r: np.ndarray, n: np.ndarray, omega: np.ndarray,
                                 Iz: float, j: float) -> float:
    """
    This computes the velocity after the collision based on the impulse resolution method
    :param r: Contact vector
    :param n: Normal of impact
    :param omega: rot velocity before impact
    :param Iz: rotational inertia
    :param j: Impulse
    :return:
    """
    return omega + np.cross(r, j * n) / Iz


def kinetic_energy(velocity: np.ndarray, m: float) -> float:
    """
    This computes the kinetic energy lost in the collision as 1/2*m*(vf^2-vi^2)
    :param velocity:   velocity right before the collision
    :param m:     mass of the object
    :return:
    """
    # todo also rotational components?
    return .5 * m * np.linalg.norm(velocity) ** 2
