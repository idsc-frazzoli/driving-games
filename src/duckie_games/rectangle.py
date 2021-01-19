from dataclasses import dataclass
from math import isclose
from decimal import Decimal as D
from itertools import product
from typing import List, Tuple, FrozenSet, Iterable
import numpy as np

from geometry import xytheta_from_SE2, SE2_from_xytheta, SE2
from zuper_commons.types import ZNotImplementedError

from games.utils import fs

from duckie_games.structures import DuckieState, DuckieGeometry, SE2_disc


Coordinates = Tuple[D, D]


@dataclass(frozen=True)
class Rectangle:
    """ Represents a rectangle """

    bottom_left: Coordinates
    top_right: Coordinates

    def sizes(self) -> Tuple[D, D]:
        return (
            self.top_right[0] - self.bottom_left[0],
            self.top_right[1] - self.bottom_left[1],
        )

    def area(self) -> D:
        a, b = self.sizes()
        return a * b

    def contains(self, c: Coordinates) -> bool:
        bl = self.bottom_left
        tr = self.top_right
        return (bl[0] <= c[0] <= tr[0]) and (bl[1] <= c[1] <= tr[1])


def get_rectangle_points_around(r: Rectangle) -> List[Coordinates]:
    bl = r.bottom_left
    tr = r.top_right

    n = 6
    res = []
    for i, j in product(range(n), range(n)):
        alpha = D(i) / D(n - 1)
        beta = D(j) / D(n - 1)
        x = bl[0] * alpha + (1 - alpha) * tr[0]
        y = bl[1] * beta + (1 - beta) * tr[1]
        res.append((D(x), D(y)))
    return res


def get_rectangle_countour(r: Rectangle) -> List[Tuple[float, float]]:
    bl = list(map(float, r.bottom_left))
    tr = list(map(float, r.top_right))
    return [
        (bl[0], bl[1]),
        (tr[0], bl[1]),
        (tr[0], tr[1]),
        (bl[0], tr[1]),
        (bl[0], bl[1]),
    ]


def make_rectangle(center: Coordinates, sides: Tuple[D, D]) -> Rectangle:
    """ Creates rectangle given center and sides. """
    c0, c1 = center
    l0, l1 = sides
    pa = c0 - l0 / 2, c1 - l1 / 2
    pb = c0 + l0 / 2, c1 + l1 / 2
    return Rectangle(pa, pb)


def sample_x(x: D, v: D, dt: D, n: int) -> List[D]:
    """Samples n points in each direction at distance dt

    For n = 2 and dt = 0.5 it will be

        -1, -0.5, 0, +0.5, +1

    """
    return [x + v * dt * i for i in range(-n, +n + 1)]


@dataclass(frozen=True)
class ProjectedCar:
    rectangle: Rectangle
    front_left: Coordinates
    front_center: Coordinates
    front_right: Coordinates


def rectangle_from_pose(ref: SE2_disc, x: D, vg: DuckieGeometry) -> ProjectedCar:
    qref = SE2_from_xytheta([float(ref[0]), float(ref[1]), np.deg2rad(float(ref[2]))])
    qd = SE2.multiply(qref, SE2_from_xytheta([float(x), 0, 0]))

    cx, cy, _ = xytheta_from_SE2(qd)
    angle_deg = ref[2]
    p = D(cx), D(cy)
    if isclose(angle_deg, 0) or isclose(angle_deg, 180) or isclose(angle_deg, -180):
        sides = (vg.length, vg.width)
        rectangle = make_rectangle(p, sides)
    elif isclose(angle_deg, 90) or isclose(angle_deg, 270):
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

    return ProjectedCar(
        rectangle,
        front_left=front_left,
        front_center=front_center,
        front_right=front_right,
    )


def get_vehicle_points(vs: DuckieState, vg: DuckieGeometry) -> FrozenSet[Coordinates]:
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


def get_resources_used(vs: DuckieState, vg: DuckieGeometry, ds: D) -> FrozenSet[Rectangle]:
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


def SE2_from_DuckieState(s: DuckieState):
    p = SE2_from_xytheta([float(s.x), 0, 0])
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    return SE2.multiply(ref, p)
