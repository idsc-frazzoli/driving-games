from dataclasses import dataclass
from decimal import Decimal as D


@dataclass
class VehicleGeometry:
    m: D
    """ Car Mass [kg] """
    w: D
    """ Car width [m] """
    l: D
    """ Half length of car - dist from CoG to each axle [m] """


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    acc: D
    """ Acceleration [m/s2] """
    dst: D
    """ Steering rate [rad/s] """

    def __add__(self, other: "VehicleActions") -> "VehicleActions":
        if type(other) == type(self):
            return VehicleActions(acc=self.acc + other.acc, dst=self.dst + other.dst)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __mul__(self, factor: D) -> "VehicleActions":
        return VehicleActions(acc=self.acc * factor, dst=self.dst * factor)

    __rmul__ = __mul__


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleState:
    x: D  # [m]
    """ CoG x location [m] """
    y: D  # [m]
    """ CoG y location [m] """
    th: D  # [rad]
    """ CoG heading [rad] """
    v: D
    """ CoG longitudinal velocity [m/s] """
    st: D
    """ Steering angle [rad] """
    t: D
    """ Time [s] """

    def __add__(self, other: "VehicleState") -> "VehicleState":
        if type(other) == type(self):
            return VehicleState(
                x=self.x + other.x,
                y=self.y + other.y,
                th=self.th + other.th,
                v=self.v + other.v,
                st=self.st + other.st,
                t=self.t + other.t,
            )
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __mul__(self, factor: D) -> "VehicleState":
        return VehicleState(
            x=self.x * factor,
            y=self.y * factor,
            th=self.th * factor,
            v=self.v * factor,
            st=self.st * factor,
            t=self.t * factor,
        )

    __rmul__ = __mul__
