from dataclasses import dataclass


@dataclass
class VehicleGeometry:
    w: float
    """ Car width [m] """
    lf: float
    """ Car length from CoG to front axle [m] """
    lr: float
    """ Car length from CoG to rear axle [m] """


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    acc: float
    """ Acceleration [m/s2] """
    dst: float
    """ Steering rate [rad/s] """

    def __add__(self, other: "VehicleActions") -> "VehicleActions":
        if type(other) == type(self):
            return VehicleActions(acc=self.acc + other.acc, dst=self.dst + other.dst)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __add__ = __add__

    def __mul__(self, factor: float) -> "VehicleActions":
        return VehicleActions(acc=self.acc * factor, dst=self.dst * factor)

    __rmul__ = __mul__


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleState:
    x: float  # [m]
    """ CoG x location [m] """
    y: float  # [m]
    """ CoG y location [m] """
    th: float  # [rad]
    """ CoG heading [rad] """
    v: float
    """ CoG longitudinal velocity [m/s] """
    st: float
    """ Steering angle [rad] """
    t: float
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

    def __mul__(self, factor: float) -> "VehicleState":
        return VehicleState(
            x=self.x * factor,
            y=self.y * factor,
            th=self.th * factor,
            v=self.v * factor,
            st=self.st * factor,
            t=self.t * factor,
        )

    __rmul__ = __mul__
