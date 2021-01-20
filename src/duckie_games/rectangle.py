from dataclasses import dataclass
from math import isclose
from decimal import Decimal as D, localcontext
from itertools import product
from typing import List, Tuple, FrozenSet, Iterable
import numpy as np

import geometry as geo
from geometry import xytheta_from_SE2, SE2_from_xytheta, SE2
from zuper_commons.types import ZNotImplementedError

from games.utils import fs

from duckie_games.structures import DuckieState, DuckieGeometry, SE2_disc
from duckie_games.utils import from_SE2_to_SE2_disc, from_SE2_disc_to_SE2


#Coordinates = Tuple[D, D]
Width = D
Height = D

class Coordinates(Tuple[D, D]):

    def __init__(self, *args, **kwargs):
        assert isinstance(self[0], D) and isinstance(self[1], D), "Inputs have to be of type Decimals"

    def as_float_tuple(self) -> Tuple[float, float]:
        """ Converts coordinates in decimals to coordinates in floats """
        return float(self[0]), float(self[1])

    @classmethod
    def from_float_tuple(cls, floats: Tuple[float, float]) -> "Coordinates":
        """ Converts coordinates in floats to coordinates in decimals"""
        coords = D(floats[0]), D(floats[1])
        return Coordinates(coords)

    def __add__(self, other):
        x = self[0] + other[0]
        y = self[1] + other[1]
        return Coordinates((x, y))

    def __sub__(self, other):
        x = self[0] - other[0]
        y = self[1] - other[1]
        return Coordinates((x, y))

    def __truediv__(self, other):
        x = self[0]
        y = self[1]
        if isinstance(other, D):
            return Coordinates((x / other, y / other))
        elif isinstance(other, float) or isinstance(other, int):
            return Coordinates((D(x) / other, D(y) / other))
        else:
            raise ZNotImplementedError()

    def __mul__(self, other):
        x = self[0]
        y = self[1]
        if isinstance(other, D):
            return Coordinates((x * other, y * other))
        elif isinstance(other, float) or isinstance(other, int):
            return Coordinates((D(x) * other, D(y) * other))
        else:
            raise ZNotImplementedError()

    __rmul__ = __mul__


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


@dataclass(frozen=True)
class RectangleWithOrientation:
    """ Represents a rectangle having orientations """

    center_pose: SE2_disc
    width: D
    height: D

    def __post_init__(self):
        check_for_type = [self.width, self.height, *self.center_pose]
        assertion = all(
            map(isinstance, check_for_type, [D] * len(check_for_type))
        )
        assert assertion, "Inputs have all to be of type Decimal"

    @property
    def sizes(self) -> Tuple[Width, Height]:
        """ Returns the width and the height of a rectangle"""
        return (
            self.width,
            self.height,
        )

    @property
    def area(self) -> D:
        """ Returns the area of the rectangle """
        a, b = self.sizes
        return a * b

    @property
    def closed_contour(self) -> List[Coordinates]:
        """
        Returns the coordinates of the closed contour of the rectangle
        The orientation of the contour is consistent with the quadrants of the rectangle reference coordinate system
                      Y
                      ^
                      |
          (1) -> x----|----x <- (0), (4)
                 |    |    |
                 |    |    |
        --------------x--------------> X
                 |    |    |
                 |    |    |
          (2) -> x----|----x <-(3)
                      |
        """

        contour_in_rect_frame = self._closed_contour_rectangle_frame  # get the rectangle contour in the rectangle frame
        contour = [self._from_rectangle_coord_to_abs(_coord) for _coord in contour_in_rect_frame]
        return contour

    @property
    def contour(self) -> List[Coordinates]:
        """
        Returns the coordinates of the contour of the rectangle.
        The orientation of the contour is consistent with the quadrants of the rectangle reference coordinate system
                      Y
                      ^
                      |
          (1) -> x----|----x <- (0)
                 |    |    |
                 |    |    |
        --------------x--------------> X
                 |    |    |
                 |    |    |
          (2) -> x----|----x <-(3)
                      |
        """

        return self.closed_contour[:-1]

    def contains_point(self, point: Coordinates, prec: int = 2) -> bool:
        """ Returns True if the point lies inside the rectangle """
        assert isinstance(point, Coordinates), "Point has to be type Coordinates"
        point_rect_frame = self._from_abs_coord_to_rectangle(point)
        return self._contains_rectangle_frame(point_rect_frame, prec=prec)

    @property
    def _closed_contour_rectangle_frame(self) -> List[Coordinates]:
        """
        Returns the coordinates of the closed contour of the rectangle in the rectangle coordinate system
        The orientation of the contour is consistent with the quadrants in the rectangle coordinate system
                      Y
                      ^
                      |
          (1) -> x----|----x <- (0), (4)
                 |    |    |
                 |    |    |
        --------------x--------------> X
                 |    |    |
                 |    |    |
          (2) -> x----|----x <-(3)
                      |
        """

        l0, l1 = self.width, self.height
        bl = - l0 / 2, - l1 / 2  # bottom left coordinate
        tr = + l0 / 2, + l1 / 2  # top right coordinate
        res = [
            (tr[0], tr[1]),
            (bl[0], tr[1]),
            (bl[0], bl[1]),
            (tr[0], bl[1]),
            (tr[0], tr[1]),
        ]
        return list(map(Coordinates, res))

    @property
    def _contour_rectangle_frame(self) -> List[Coordinates]:
        """
        Returns the coordinates of the closed contour of the rectangle in the rectangle coordinate system
        The orientation of the contour is consistent with the quadrants in the rectangle coordinate system
                      Y
                      ^
                      |
          (1) -> x----|----x <- (0), (4)
                 |    |    |
                 |    |    |
        --------------x--------------> X
                 |    |    |
                 |    |    |
          (2) -> x----|----x <-(3)
                      |
        """

        return self._closed_contour_rectangle_frame[:-1]

    @property
    def _top_right(self) -> Coordinates:
        """ Returns the top right corner in the rectangle coordinate system """
        return self._closed_contour_rectangle_frame[0]

    @property
    def _bottom_left(self) -> Coordinates:
        """ Returns the bottom left coordinates in the rectangle coordinate system """
        return self._closed_contour_rectangle_frame[2]

    def _contains_rectangle_frame(self, coord: Coordinates, prec: int = 2) -> bool:
        """ Returns True if the coordinates (rectangle coordinate system) lie inside the rectangle """
        with localcontext() as ctx:
            ctx.prec = prec
            bl = self._bottom_left * 1  # get the rounded values (multiplying by 1)
            tr = self._top_right * 1
            c = coord * 1
        res = (bl[0] <= c[0] <= tr[0]) and (bl[1] <= c[1] <= tr[1])
        return res

    def _from_rectangle_coord_to_abs(self, coord: Coordinates) -> Coordinates:
        """
        Returns the coordinates in the absolute coordinate system
        """
        center_pose_SE2 = from_SE2_disc_to_SE2(self.center_pose)  # get the pose of the rectangle as SE2
        # convert the coordinates to float tuples
        coord_as_floats = coord.as_float_tuple()

        # get the coordinates in the absolute frame
        coord_abs_as_floats = geo.SE2.multiply(  # apply change of coordinates
            center_pose_SE2,  # pose of the rectangle
            (*coord_as_floats, 1)  # coordinates in homogenous representation
        )[:-1]  # transform back to Euclidean representation
        coord_abs = Coordinates.from_float_tuple(coord_abs_as_floats)  # transform back to coordinates
        return coord_abs

    def _from_abs_coord_to_rectangle(self, coord: Coordinates) -> Coordinates:
        """
        Returns the coordinates in the rectangle coordinate system
        """
        center_pose_SE2 = from_SE2_disc_to_SE2(self.center_pose)  # get the pose of the rectangle as SE2
        center_pose_SE2_inv = geo.SE2.inverse(center_pose_SE2)
        # convert the coordinates to float tuples
        coord_as_floats = coord.as_float_tuple()

        # get the coordinates in the absolute frame
        coord_abs_as_floats = geo.SE2.multiply(  # apply change of coordinates
            center_pose_SE2_inv,  # inverse pose of the rectangle
            (*coord_as_floats, 1)  # coordinates in homogenous representation
        )[:-1]  # transform back to Euclidean representation
        coord_abs = Coordinates.from_float_tuple(coord_abs_as_floats)  # transform back to coordinates
        return coord_abs




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


def projected_car_from_state(x: DuckieState, vg: DuckieGeometry) -> ProjectedCar:
    length = vg.length
    width = vg.width
    rect = RectangleWithOrientation(
        center_pose=x.abs_pose,
        width=width,
        height=length
    )
    rect_contour = rect.contour
    front_left = rect_contour[1]
    front_right = rect_contour[0]
    front_center = front_left + (front_right - front_left) / 2

    return ProjectedCar(
        rectangle=rect,
        front_left=front_left,
        front_center=front_center,
        front_right=front_right
    )

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
