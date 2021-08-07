import math
from dataclasses import dataclass, replace

import numpy as np
from frozendict import frozendict
from geometry import T2value, SO2_from_angle, SO2value

from sim.models.utils import kmh2ms
from sim.models.vehicle import VehicleCommands, VehicleState, VehicleModel
from sim.models.vehicle_structures import VehicleParameters, VehicleGeometry


@dataclass(frozen=True, unsafe_hash=True)
class VehicleParametersDyn(VehicleParameters):
    """Provides additional parameters with respect to the kinematic model"""

    @classmethod
    def default_car(cls) -> "VehicleParametersDyn":
        return VehicleParametersDyn(vx_limits=(kmh2ms(-10), kmh2ms(130)),
                                    acc_limits=(-8, 5),
                                    delta_max=math.pi / 2,
                                    ddelta_max=15)

    @classmethod
    def default_bicycle(cls) -> "VehicleParametersDyn":
        return VehicleParametersDyn(vx_limits=(kmh2ms(-1), kmh2ms(50)),
                                    acc_limits=(-4, 3),
                                    delta_max=math.pi / 4,
                                    ddelta_max=5)

    def __post_init__(self):
        super().__post_init__()


class VehicleTires:
    # todo
    pass


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleStateDyn(VehicleState):
    vy: float
    """ CoG longitudinal velocity [m/s] """
    dtheta: float
    """ yaw rate """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3, "vy": 4, "dtheta": 5, "delta": 6})
    """ Dictionary to get correct values from numpy arrays"""

    def __add__(self, other: "VehicleStateDyn") -> "VehicleStateDyn":
        if type(other) == type(self):
            return replace(self,
                           x=self.x + other.x,
                           y=self.y + other.y,
                           theta=self.theta + other.theta,
                           vx=self.vx + other.vx,
                           vy=self.vy + other.vy,
                           dtheta=self.dtheta + other.dtheta,
                           delta=self.delta + other.delta
                           )
        else:
            raise NotImplementedError

    def __mul__(self, val: float) -> "VehicleStateDyn":
        return replace(self,
                       x=self.x * val,
                       y=self.y * val,
                       theta=self.theta * val,
                       vx=self.vx * val,
                       vy=self.vy * val,
                       dtheta=self.dtheta * val,
                       delta=self.delta * val,
                       )

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta,
            self.vx,
            self.vy,
            self.dtheta,
            self.delta
        ])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return VehicleStateDyn(x=z[cls.idx["x"]],
                               y=z[cls.idx["y"]],
                               theta=z[cls.idx["theta"]],
                               vx=z[cls.idx["vx"]],
                               vy=z[cls.idx["vy"]],
                               dtheta=z[cls.idx["dtheta"]],
                               delta=z[cls.idx["delta"]])


class VehicleModelDyn(VehicleModel):

    def __init__(self, x0: VehicleStateDyn, vg: VehicleGeometry, vp: VehicleParametersDyn, vtires: VehicleTires = None):
        """

        :param x0:
        :param vg:
        :param vp:
        :param vtires:
        """
        super(VehicleModelDyn, self).__init__(x0, vg, vp)
        # """ The vehicle's geometry parameters"""
        self.vp: VehicleParametersDyn = vp
        """ The vehicle parameters"""
        self.vtires: VehicleTires = vtires
        """ The vehicle tyre model"""

    @classmethod
    def default_bicycle(cls, x0: VehicleStateDyn):
        return VehicleModelDyn(x0=x0, vg=VehicleGeometry.default_bicycle(), vp=VehicleParametersDyn.default_bicycle())

    @classmethod
    def default_car(cls, x0: VehicleStateDyn):
        return VehicleModelDyn(x0=x0, vg=VehicleGeometry.default_car(), vp=VehicleParametersDyn.default_car())

    def dynamics(self, x0: VehicleStateDyn, u: VehicleCommands, mean: bool = True) -> VehicleStateDyn:
        """ returns state derivative for given control inputs """

        print(f"the right call with state {x0},commands {u}")

        return x0

    def get_velocity(self, in_model_frame: bool = True) -> (T2value, float):
        v_l = np.array([self._state.vx, self._state.vy])
        if in_model_frame:
            return v_l, self._state.dtheta
        rot: SO2value = SO2_from_angle(self._state.theta)
        v_g = rot @ v_l
        return v_g, self._state.dtheta

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool = True):
        # fixme inconsistent local/global ref frame
        if in_model_frame:
            self._state.vx = vel[0]
            self._state.vy = vel[1]
            self._state.dtheta = omega
        else:
            raise NotImplementedError
