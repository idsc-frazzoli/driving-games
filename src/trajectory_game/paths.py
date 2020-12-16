from abc import ABCMeta, abstractmethod
from numbers import Real
from typing import Tuple, List, TypeVar, Generic, ClassVar, Type, Iterable, Union
from decimal import Decimal as D
from math import atan2
import numpy as np
from scipy import interpolate

from .sequence import SampledSequence, Timestamp
from .structures import VehicleState

__all__ = [
    "Curve",
    "SplineCurve",
    "Path",
    "PathWithBounds",
    "SplinePath",
    "SplinePathWithBounds",
    "Trajectory",
]

X = TypeVar("X")
ArrayLike = Union[np.ndarray, Iterable, int, float]


class Curve(Generic[X], metaclass=ABCMeta):
    """ Base class for 1d curves. """

    XT: ClassVar[Type[X]] = object

    @abstractmethod
    def value_at_s(self, s: ArrayLike) -> List[X]:
        """ Calculate value of curve at requested coordinates """

    @abstractmethod
    def derivative_at_s(self, s: ArrayLike) -> List[X]:
        """ Calculate derivatives of curve at requested coordinates """


class SplineCurve(Curve[X]):
    """ Line class for linear interpolation of a curve. """

    s: List[X]
    """ Abscissa for interpolation of the curve. """
    z: List[X]
    """ Values of the curve at abscissa points """
    _tck: Tuple[np.ndarray, np.ndarray, int]
    """ Spline coefficients - generated automatically. """

    def __init__(self, s: List[Real], z: List[X], order: int):
        if len(s) != len(z):
            msg = "InterpCurve Length mismatch"
            raise ValueError(msg)

        for i in range(len(s) - 1):
            ds = s[i + 1] - s[i]
            if ds <= 0:
                msg = "Invalid ds = %s at i = %s; ts= %s" % (ds, i, s)
                raise ValueError(msg)

        self.s = s
        self.z = z
        self._tck = interpolate.splrep(x=s, y=z, k=order)

    def value_at_s(self, s: ArrayLike) -> List[X]:
        """ Calculate value of curve at requested values """
        values = interpolate.splev(x=s, tck=self._tck, der=0, ext=2)
        return [D(_) for _ in values]

    def derivative_at_s(self, s: ArrayLike, order: int = 1) -> List[X]:
        """Calculate derivatives of curve at requested values
        Default is first derivative, but any order (<k) can be calculated"""
        if order >= self._tck[-1]:
            msg = "Max order for derivative is {}, requested {}".format(self._tck[-1] - 1, order)
            raise ValueError(msg)
        if order <= 0:
            msg = "Derivative order needs to be positive, requested {}".format(order)
            raise ValueError(msg)
        ret = interpolate.splev(x=s, tck=self._tck, der=order, ext=2)
        return [D(_) for _ in ret]

    def get_end(self) -> Timestamp:
        return D(float(self.s[-1]))


class Path(Generic[X], metaclass=ABCMeta):
    """ Base class for all 2d transitions. """

    @abstractmethod
    def get_s_limits(self) -> Tuple[D, D]:
        """ Returns progress limits of reference """

    @abstractmethod
    def value_at_s(self, s: ArrayLike) -> List[Tuple[X, X]]:
        """ Calculate [x,y] of transition at progress values """

    @abstractmethod
    def heading_at_s(self, s: ArrayLike) -> List[X]:
        """ Calculate heading of transition at progress values """

    @abstractmethod
    def cartesian_to_curvilinear(self, xy: List[Tuple[X, X]]) -> List[Tuple[D, X]]:
        """ Converts cartesian coordinates [x,y] to curvilinear [s,n] """

    @abstractmethod
    def curvilinear_to_cartesian(self, sn: List[Tuple[D, X]]) -> List[Tuple[X, X]]:
        """ Converts curvilinear coordinates [s,n] to cartesian [x,y] """


class PathWithBounds(Path[X], metaclass=ABCMeta):
    """ Base class for reference paths with lane bounds """

    @abstractmethod
    def get_bounds_at_s(self, s: List[Real]) -> List[Tuple[D, D]]:
        """ Return left and right boundaries in curvilinear coordinates at progress """


class SplinePath(Path[X]):
    """ Path defined by splines. """

    s: List[Real]
    """ Abscissa for interpolation of the curve. """
    x: SplineCurve[X]
    """ Curve for x-coordinate. """
    y: SplineCurve[X]
    """ Curve for y-coordinate. """
    points: SampledSequence[Tuple[X, X]]
    """ Sampled [x,y] points vs progress. """
    deriv: SampledSequence[Tuple[X, X]]
    """ Sampled [x,y] derivatives at points vs progress. """

    def __init__(self, s: List[Real], x: List[X], y: List[X], order: int):
        self.x = SplineCurve(s=s, z=x, order=order)
        self.y = SplineCurve(s=s, z=y, order=order)
        self.s = s

        # TODO[SIR]: Any better way than casting and recasting for numpy?
        # def cast_list(input: List[float]) -> List[Real]:
        #     return [D(_) for _ in input]

        step: float = 0.1
        # N: int = int((s[-1] - s[0]) // step)
        p_s: ArrayLike = np.arange(float(s[0]), float(s[-1]), step)
        p_x = self.x.value_at_s(p_s)
        p_y = self.y.value_at_s(p_s)
        p_xy = list(zip(p_x, p_y))
        p_s_D: List[D] = [D(p) for p in p_s]
        self.points = SampledSequence(timestamps=p_s_D, values=p_xy)
        d_x = self.x.derivative_at_s(p_s, order=1)
        d_y = self.y.derivative_at_s(p_s, order=1)
        d_xy = list(zip(d_x, d_y))
        self.deriv = SampledSequence(timestamps=p_s_D, values=d_xy)

    def get_s_limits(self) -> Tuple[Real, Real]:
        return self.s[0], self.s[-1]

    def value_at_s(self, s: List[Real]) -> List[Tuple[X, X]]:
        x = self.x.value_at_s(s)
        y = self.y.value_at_s(s)
        ret = list(zip(x, y))
        return ret

    def heading_at_s(self, s: List[Real]) -> List[X]:
        dx = self.x.derivative_at_s(s, order=1)
        dy = self.y.derivative_at_s(s, order=1)

        def calc_heading(x: X, y: X) -> D:
            return D(atan2(y, x))

        ret = [calc_heading(x, y) for x, y in zip(dx, dy)]
        return ret

    def cartesian_to_curvilinear(self, xy: List[Tuple[X, X]]) -> List[Tuple[D, X]]:
        ret = []

        def get_n(p1: Tuple[X, X], p2: Tuple[X, X]) -> X:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2).sqrt()

        def get_n_sign(s_p: D, xy_p: Tuple[X, X]) -> int:
            p_s = self.points.at(s_p)
            d_s = self.deriv.at(s_p)
            dx = xy_p[0] - p_s[0]
            dy = xy_p[1] - p_s[1]
            sign: int = -1 + 2 * ((d_s[0] * dy - d_s[1] * dx) > 0)
            return sign

        for p_xy in xy:
            best_s: D = D("0")
            best_n: D = D("1000")
            for s, p in self.points:
                n = get_n(p_xy, p)
                if n < best_n:
                    best_s = s
                    best_n = n
            ret.append((best_s, best_n * get_n_sign(best_s, p_xy)))

        return ret

    def curvilinear_to_cartesian(self, sn: List[Tuple[D, X]]) -> List[Tuple[X, X]]:
        """ Converts curvilinear coordinates [s,n] to cartesian [x,y] """
        pass


class SplinePathWithBounds(PathWithBounds[X], SplinePath[X]):
    """ Spline reference path with lane bounds """

    left: Curve[D]
    """ Left lane boundary """
    right: Curve[D]
    """ Right lane boundary """

    def __init__(
        self,
        s: List[Real],
        p_ref: List[Tuple[D, D]],
        p_left: List[Tuple[D, D]],
        p_right: List[Tuple[D, D]],
        bounds_sn: bool = False,
    ):
        x, y = zip(*p_ref)
        super().__init__(s, x, y, order=3)

        def fit_curve(p: List[Tuple[D, D]]) -> Curve[D]:
            sn: List[Tuple[D, D]] = p if bounds_sn else self.cartesian_to_curvilinear(p)
            p_s, n = list(zip(*sn))
            curve = SplineCurve(s=p_s, z=n, order=3)
            return curve

        self.left = fit_curve(p_left)
        self.right = fit_curve(p_right)

    def get_bounds_at_s(self, s: List[Real]) -> List[Tuple[D, D]]:
        """ Return left and right boundaries in curvilinear coordinates at progress """
        left = self.left.value_at_s(s)
        right = self.right.value_at_s(s)
        ret = list(zip(right, left))
        return ret


class Trajectory:
    """ Container for trajectory - path + velocities, steering """

    traj: SampledSequence[VehicleState]

    def __init__(self, traj: List[VehicleState]):
        times: List[Timestamp] = [t.t for t in traj]
        self.traj = SampledSequence(timestamps=times, values=traj)

    def get_sequence(self) -> SampledSequence[VehicleState]:
        """ Returns sequence of trajectory points """
        return self.traj

    def get_sampling_points(self) -> List[Timestamp]:
        """ Returns timestamps of trajectory points """
        return self.traj.get_sampling_points()

    def get_path(self) -> List[Tuple[D, D]]:
        """ Returns cartesian coordinates [x,y] of trajectory """
        ret = [(x.x, x.y) for _, x in self.traj]
        return ret

    def __iter__(self):
        return self.traj.__iter__()
