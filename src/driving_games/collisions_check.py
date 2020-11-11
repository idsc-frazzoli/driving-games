from dataclasses import dataclass
from decimal import Decimal as D
from typing import FrozenSet, Iterable, List, Mapping, Tuple

import numpy as np
from frozendict import frozendict

from games import PlayerName
from games.utils import fs
from geometry import SE2, SE2_from_xytheta, xytheta_from_SE2
from zuper_commons.types import ZNotImplementedError
from .collisions import Collision, IMPACT_FRONT, IMPACT_SIDES, ProjectedCar
from .rectangle import (
    Coordinates,
    get_rectangle_points_around,
    make_rectangle,
    Rectangle,
)
from .structures import SE2_disc, VehicleGeometry, VehicleState

__all__ = ["collision_check"]


# XXX: Note that this only works for the simplest cases.
#      For example it does not work for head-to-back collision.
def collision_check(
    poses: Mapping[PlayerName, VehicleState], geometries: Mapping[PlayerName, VehicleGeometry],
) -> Mapping[PlayerName, Collision]:
    dt = D(0.5)
    n = 2
    if len(poses) == 1:
        return frozendict({})
    if len(poses) > 2:
        raise ZNotImplementedError(players=set(poses))

    p1, p2 = list(poses)
    s1 = poses[p1]
    s2 = poses[p2]
    g1 = geometries[p1]
    g2 = geometries[p2]

    x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
    x2s = sample_x(s2.x, s2.v, dt=dt, n=n)

    for x1, x2 in zip(x1s, x2s):
        pc1 = rectangle_from_pose(s1.ref, x1, g1)
        pc2 = rectangle_from_pose(s2.ref, x2, g2)

        # did p1 collide with p2?
        p1_caused = a_caused_collision_with_b(pc1, pc2)
        p2_caused = a_caused_collision_with_b(pc2, pc1)

        p1_active = p1_caused
        p2_active = p2_caused
        if p1_caused and p2_caused:
            # head-on collision
            i1 = i2 = IMPACT_FRONT
            vs = s1.v * g1.mass + s2.v * g2.mass
            energy_received_1 = vs
            energy_received_2 = vs
            energy_given_1 = vs
            energy_given_2 = vs
            pass
        elif p1_caused:
            i1 = IMPACT_FRONT
            i2 = IMPACT_SIDES
            energy_received_1 = D(0)
            energy_received_2 = s1.v * g1.mass
            energy_given_1 = s1.v * g1.mass
            energy_given_2 = D(0)
        elif p2_caused:
            i1 = IMPACT_SIDES
            i2 = IMPACT_FRONT
            energy_received_2 = D(0)
            energy_received_1 = s1.v * g1.mass
            energy_given_2 = s1.v * g1.mass
            energy_given_1 = D(0)
        else:
            continue

        c1 = Collision(i1, p1_active, energy_received_1, energy_given_1)
        c2 = Collision(i2, p2_active, energy_received_2, energy_given_2)
        return frozendict({p1: c1, p2: c2})

    return frozendict({})


def a_caused_collision_with_b(a: ProjectedCar, b: ProjectedCar):
    return any(b.rectangle.contains(_) for _ in (a.front_right, a.front_center, a.front_left))


def sample_x(x: D, v: D, dt: D, n: int) -> List[D]:
    """ Samples n points in each direction at distance dt

        For n = 2 and dt = 0.5 it will be

            -1, -0.5, 0, +0.5, +1

    """
    return [x + v * dt * i for i in range(-n, +n + 1)]


def rectangle_from_pose(ref: SE2_disc, x: D, vg: VehicleGeometry) -> ProjectedCar:
    qref = SE2_from_xytheta([float(ref[0]), float(ref[1]), np.deg2rad(float(ref[2]))])
    qd = SE2.multiply(qref, SE2_from_xytheta([float(x), 0, 0]))

    cx, cy, _ = xytheta_from_SE2(qd)
    angle_deg = ref[2]
    p = D(cx), D(cy)
    if angle_deg == 0 or angle_deg == 180 or angle_deg == -180:
        sides = (vg.length, vg.width)
        rectangle = make_rectangle(p, sides)
    elif angle_deg == 90 or angle_deg == 270:
        sides = (vg.width, vg.length)
        rectangle = make_rectangle(p, sides)
    else:
        raise ZNotImplementedError(p=p, angle_deg=angle_deg, vg=vg)

    L = float(vg.length)
    W = float(vg.width)
    front_left_b = (L / 2, W / 2, 1)
    front_center_b = (L / 2, 0, 1)
    front_right_b = (L / 2, -W / 2, 1)

    front_left = (qd @ front_left_b)[:2]
    front_center = (qd @ front_center_b)[:2]
    front_right = (qd @ front_right_b)[:2]

    return ProjectedCar(rectangle, front_left=front_left, front_center=front_center, front_right=front_right,)


@dataclass
class SpaceTimeCell:
    """ A chunk of space time """

    space: Rectangle
    time: Tuple[D, D]


def get_vehicle_points(vs: VehicleState, vg: VehicleGeometry) -> FrozenSet[Coordinates]:
    """ Gets a set of representative points for the vehicle"""
    # this will extend -0.5 0.25 0 0.25 0.5
    dt = D(0.25)
    n = 2
    xs = sample_x(vs.x, vs.v, dt=dt, n=n)

    points = set()
    for x in xs:
        r = rectangle_from_pose(vs.ref, x, vg).rectangle
        rp = get_rectangle_points_around(r)
        points.update(rp)

    return fs(points)


def get_resources_used(vs: VehicleState, vg: VehicleGeometry, ds: D) -> FrozenSet[Rectangle]:
    """ Gets the rectangles that contain the vehicle. """
    points = get_vehicle_points(vs, vg)

    rectangles = rectangles_from_points(points, ds)
    return rectangles


def rectangles_from_points(points: Iterable[Coordinates], ds: D) -> FrozenSet[Rectangle]:
    """ Gets the rectangles containing all these coordinates. """
    return frozenset(rectangle_from_point(_, ds) for _ in points)


def rectangle_from_point(point: Coordinates, ds: D) -> Rectangle:
    """ Gets the rectangle containing this coordinate. """
    x = int(point[0] / ds) * ds
    y = int(point[1] / ds) * ds

    bottom_left = (x, y)
    top_right = (x + ds, y + ds)
    return Rectangle(bottom_left=bottom_left, top_right=top_right)
