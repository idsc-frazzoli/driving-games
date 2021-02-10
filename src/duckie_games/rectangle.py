import math

from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from itertools import product
from typing import List, Tuple, FrozenSet, Iterable
import numpy as np

import geometry as geo
from geometry import SE2_from_xytheta, SE2
from zuper_commons.types import ZNotImplementedError

from games.utils import fs

from duckie_games.structures import DuckieState, DuckieGeometry, SE2_disc
from world.utils import (
    from_SE2_disc_to_SE2,
    Lane,
    get_SE2disc_from_along_lane
)


Width = D
Height = D

Angle_Deg = D
Angle_Rad = D


class Coordinates(Tuple[D, D]):
    """
    Class representing 2D Coordinates in Decimals. Support simple addition and subtraction as well as
    multiplication and division by a scalar.
    """
    def __init__(self, *args, **kwargs):
        assert isinstance(self[0], D) and isinstance(self[1], D), "Inputs have to be of type Decimals"

    def as_float_tuple(self) -> Tuple[float, float]:
        """ Converts coordinates in decimals to coordinates in floats """
        return float(self[0]), float(self[1])

    def as_polar(self) -> Tuple[D, Angle_Rad]:
        """
        Returns the coordinates as polar coordinates.
        Theta is always between ]-180°, 180°]
        """
        x = self[0]
        y = self[1]
        r_squared = pow(x, 2) + pow(y, 2)
        r = r_squared.sqrt()
        theta_deg_float = np.rad2deg(math.atan2(y, x))
        theta_deg = D(theta_deg_float)
        return r, theta_deg

    @classmethod
    def from_float_tuple(cls, floats: Tuple[float, float]) -> "Coordinates":
        """ Converts coordinates in floats to coordinates in decimals"""
        coords = D(floats[0]), D(floats[1])
        return Coordinates(coords)

    def __add__(self, other):
        if isinstance(other, Coordinates):
            x = self[0] + other[0]
            y = self[1] + other[1]
            return Coordinates((x, y))
        else:
            raise ZNotImplementedError("Must be of type Coordinates")

    def __sub__(self, other):
        if isinstance(other, Coordinates):
            x = self[0] - other[0]
            y = self[1] - other[1]
            return Coordinates((x, y))
        else:
            raise ZNotImplementedError("Must be of type Coordinates")

    def __truediv__(self, other):
        x = self[0]
        y = self[1]
        if isinstance(other, D):
            return Coordinates((x / other, y / other))
        elif isinstance(other, float) or isinstance(other, int):
            return Coordinates((x / D(other), y / D(other)))
        else:
            raise ZNotImplementedError("Division of those types not supported")

    def __mul__(self, other):
        x = self[0]
        y = self[1]
        if isinstance(other, D):
            return Coordinates((x * other, y * other))
        elif isinstance(other, float) or isinstance(other, int):
            return Coordinates((x * D(other), y * D(other)))
        else:
            raise ZNotImplementedError("Multiplication of those types not supported")

    __rmul__ = __mul__


@dataclass(frozen=True)
class Rectangle:
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
    def center(self) -> Coordinates:
        """ Returns the coordinates of the center of the rectangle """
        return Coordinates(
            (self.center_pose[0], self.center_pose[1])
        )

    @property
    def orientation_in_deg(self) -> Angle_Deg:
        """ Returns the orientation of the rectangle in degrees """
        return self.center_pose[2]

    @property
    def orientation_in_rad(self) -> Angle_Rad:
        """ Returns the orientation of the rectangle in rads """
        orient_deg = self.orientation_in_deg
        orient_rad = D(
            np.deg2rad(float(orient_deg))
        )
        return orient_rad

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
        """
        Returns True if the point lies inside the rectangle
        prec: precisions of decimals during comparison
        """
        assert isinstance(point, Coordinates), "Point has to be type Coordinates"
        point_rect_frame = self._from_abs_coord_to_rectangle(point)
        return self._contains_rectangle_frame(point_rect_frame, prec=prec)

    def get_points_inside(self, n: int = 6) -> List[Coordinates]:
        """
        Returns evenly spaced points inside the rectangle.
        n points along width and n points along height.
        """
        bl = self._bottom_left
        tr = self._top_right

        res: List[Coordinates] = []
        for i, j in product(range(n), range(n)):
            alpha = D(i) / D(n - 1)
            beta = D(j) / D(n - 1)
            x = bl[0] * alpha + (1 - alpha) * tr[0]
            y = bl[1] * beta + (1 - beta) * tr[1]
            coord_rect_frame = Coordinates((D(x), D(y)))
            coords = self._from_rectangle_coord_to_abs(coord_rect_frame)
            res.append(coords)
        return res

    @property
    def angles_diagnoals(self) -> Tuple[Angle_Deg, Angle_Deg]:
        """
        Returns the angles between the diagonals
                      Y   angle_y
                      ^  /
                      | v
                  x---|---x
                  | \ |  /|
                  |  \| / | <- angle_x
        --------------x--------------> X
                  |  /| \ |
                  | / |  \|
                  x----|---x
                      |
        """
        top_right = self._closed_contour_rectangle_frame[0]
        top_left = self._closed_contour_rectangle_frame[1]
        _, angle_top_right = top_right.as_polar()
        _, angle_top_left = top_left.as_polar()
        angle_y = angle_top_left - angle_top_right
        angle_x = D(180) - angle_y
        return angle_x, angle_y

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
        """
        Returns True if the coordinates (rectangle coordinate system) lie inside the rectangle
        prec: precisions of decimals during comparison
        """
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


def sample_x_speed_dep(x: D, v: D, dt: D, n_min: int, n_max: int) -> List[D]:
    """Samples n points from last timestep next timestep depending on speed

        """
    v_max = 6  # todo
    v_min = 0
    n = int(n_min + (v - v_min) / (v_max - v_min) * (n_max - n_min))
    return [x + v * dt * round(D(i / n), 2) for i in range(-n, n+1)]


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


def projected_car_from_along_lane(lane: Lane, along_lane: D, vg: DuckieGeometry) -> ProjectedCar:
    """
    Generates a car represented as a rectangle. With orientation 0° the car drives from left to right,
     therefore is the length of the car equivalent to the width of the rectangle.
    """
    center_pose = get_SE2disc_from_along_lane(lane=lane, along_lane=along_lane)
    width = vg.length
    length = vg.width
    rect = Rectangle(
        center_pose=center_pose,
        width=width,
        height=length
    )
    rect_contour: List[Coordinates] = rect.contour
    front_left = rect_contour[0]
    front_right = rect_contour[-1]
    front_center = front_left + (front_right - front_left) / 2

    return ProjectedCar(
        rectangle=rect,
        front_left=front_left,
        front_center=front_center,
        front_right=front_right
    )


def projected_car_from_state(x: DuckieState, vg: DuckieGeometry) -> ProjectedCar:
    """
    Generates a car represented as a rectangle. With orientation 0° the car drives from left to right,
     therefore is the length of the car equivalent to the width of the rectangle.
    """
    lane = x.lane
    along_lane = x.x
    return projected_car_from_along_lane(
        lane=lane,
        along_lane=along_lane,
        vg=vg
    )


def get_vehicle_points(vs: DuckieState, vg: DuckieGeometry) -> FrozenSet[Coordinates]:
    """ Gets a set of representative points for the vehicle"""
    # this will extend -0.5 0.25 0 0.25 0.5
    dt = D(0.25)
    n = 2
    xs = sample_x(vs.x, vs.v, dt=dt, n=n)

    points = set()
    for x in xs:
        r = projected_car_from_along_lane(lane=vs.lane, along_lane=x, vg=vg).rectangle
        rp = r.get_points_inside()
        points.update(rp)

    return fs(points)


def get_resources_used_old(vs: DuckieState, vg: DuckieGeometry, ds: D) -> FrozenSet[Rectangle]:
    """ Gets the rectangles that contain the vehicle. """
    points = get_vehicle_points(vs, vg)
    orient = vs.abs_pose[2]  # orientation of vehicle
    rectangles = rectangles_from_points(points, orient, ds)
    return rectangles


def rectangles_from_points(points: Iterable[Coordinates], orient: D,  ds: D) -> FrozenSet[Rectangle]:
    """ Gets the rectangles containing all these coordinates. """
    return frozenset(rectangle_from_point(_, orient, ds) for _ in points)


def rectangle_from_point(point: Coordinates, orient: D, ds: D) -> Rectangle:
    """ Gets the rectangle containing this coordinate. """
    pose = (point[0], point[1], orient)
    return Rectangle(center_pose=pose, width=ds, height=ds)


def SE2_from_DuckieState(s: DuckieState):
    p = SE2_from_xytheta([float(s.x), 0, 0])
    ref = SE2_from_xytheta([float(s.ref[0]), float(s.ref[1]), np.deg2rad(float(s.ref[2]))])
    return SE2.multiply(ref, p)


def two_rectangle_intersection(r1: Rectangle, r2: Rectangle) -> bool:
    """
    Returns True if the two rectangles intersect
    """
    r1_test_p = r1.contour + [r1.center]
    r2_test_p = r2.contour + [r2.center]
    r1_cont_in_r2 = any(r2.contains_point(p) for p in r1_test_p)
    r2_cont_in_r1 = any(r1.contains_point(p) for p in r2_test_p)

    return r1_cont_in_r2 or r2_cont_in_r1
