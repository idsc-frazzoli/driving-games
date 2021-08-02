from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal

import math
from functools import cached_property
from typing import Tuple

import numpy as np
from commonroad_dc.pycrcc import RectOBB

from frozendict import frozendict
from geometry import SE2value, SE2_from_xytheta
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon

from sim.models.utils import kmh2ms
from sim.simulator_structures import SimModel
from sim.typing import Color


@dataclass(frozen=True, unsafe_hash=True)
class ModelGeometry(ABC):

    @property
    @abstractmethod
    def outline(self):
        pass


@dataclass(frozen=True, unsafe_hash=True)
class VehicleGeometry(ModelGeometry):
    """ Geometry parameters of the vehicle (and colour)"""

    m: float
    """ Vehicle Mass [kg] """
    w_half: float
    """ Half width of vehicle [m] """
    l_half: float
    """ Half length of vehicle - dist from CoG to each axle [m] """
    color: Color = (1, 1, 1)
    """ Color """

    @classmethod
    def default_car(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=1000.0, w_half=1.0, l_half=2.0)

    @classmethod
    def default_bicycle(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=80.0, w_half=0.25, l_half=1.0)

    @cached_property
    def width(self):
        return self.w_half * 2

    @cached_property
    def length(self):
        return self.l_half * 2

    @cached_property
    def outline(self) -> Tuple[Tuple[float, float], ...]:
        return ((-self.l_half, -self.w_half), (-self.l_half, +self.w_half),
                (+self.l_half, +self.w_half), (+self.l_half, -self.w_half), (-self.l_half, -self.w_half))


@dataclass(frozen=True, unsafe_hash=True)
class VehicleParameters:
    vx_max: float
    vx_min: float
    """ Maximum and Minimum velocities [m/s] """
    delta_max: float
    """ Maximum steering angle [rad] """

    @classmethod
    def default_car(cls) -> "VehicleParameters":
        return VehicleParameters(vx_max=kmh2ms(130), vx_min=kmh2ms(-10), delta_max=math.pi)

    @classmethod
    def default_bicycle(cls) -> "VehicleParameters":
        return VehicleParameters(vx_max=kmh2ms(50), vx_min=kmh2ms(-1), delta_max=math.pi)


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleCommands:
    acc: float
    """ Acceleration [m/s^2] """
    ddelta: float
    """ Steering rate [rad/s] (delta derivative) """
    idx = frozendict({"acc": 0, "ddelta": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "VehicleCommands") -> "VehicleCommands":
        if type(other) == type(self):
            return replace(self,
                           acc=self.acc + other.acc,
                           ddelta=self.ddelta + other.ddelta)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleCommands") -> "VehicleCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "VehicleCommands":
        return replace(self, acc=self.acc * val, ddelta=self.ddelta * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "VehicleCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.acc,
            self.ddelta
        ])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return VehicleCommands(acc=z[cls.idx["acc"]],
                               ddelta=z[cls.idx["ddelta"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleState:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    theta: float
    """ CoG heading [rad] """
    vx: float
    """ CoG longitudinal velocity [m/s] """
    delta: float
    """ Steering angle [rad] """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3, "delta": 4})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "VehicleState") -> "VehicleState":
        if type(other) == type(self):
            return replace(self,
                           x=self.x + other.x,
                           y=self.y + other.y,
                           theta=self.theta + other.theta,
                           vx=self.vx + other.vx,
                           delta=self.delta + other.delta
                           )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleState") -> "VehicleState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "VehicleState":
        return replace(self,
                       x=self.x * val,
                       y=self.y * val,
                       theta=self.theta * val,
                       vx=self.vx * val,
                       delta=self.delta * val,
                       )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "VehicleState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta,
            self.vx,
            self.delta
        ])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return VehicleState(x=z[cls.idx["x"]],
                            y=z[cls.idx["y"]],
                            theta=z[cls.idx["theta"]],
                            vx=z[cls.idx["vx"]],
                            delta=z[cls.idx["delta"]])


class VehicleModel(SimModel[VehicleState, VehicleCommands]):
    # fixme make sure class attributes don't mess with instance once
    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""
    vp: VehicleParameters
    """ The vehicle parameters"""
    _state: VehicleState
    """ Current state of the model"""

    def __init__(self, x0: VehicleState, vg: VehicleGeometry, vp: VehicleParameters):
        self._state = x0
        self.vg = vg
        self.vp = vp

    @classmethod
    def default_bicycle(cls, x0: VehicleState):
        return VehicleModel(x0=x0, vg=VehicleGeometry.default_bicycle(), vp=VehicleParameters.default_bicycle())

    @classmethod
    def default_car(cls, x0: VehicleState):
        return VehicleModel(x0=x0, vg=VehicleGeometry.default_car(), vp=VehicleParameters.default_car())

    def update(self, commands: VehicleCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [VehicleState, VehicleCommands]:
            n_states = VehicleState.get_n_states()
            state = VehicleState.from_array(y[0:n_states])
            actions = VehicleCommands(acc=y[VehicleCommands.idx["acc"] + n_states],
                                      ddelta=y[VehicleCommands.idx["ddelta"] + n_states])
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions, mean=False)
            du = np.zeros([len(VehicleCommands.idx)])
            return np.concatenate([dx.as_ndarray(), du])

        state_np = self._state.as_ndarray()
        action_np = commands.as_ndarray()
        y0 = np.concatenate([state_np, action_np])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(dt)), y0=y0)

        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")
        new_state, _ = _stateactions_from_array(result.y[:, -1])
        self._state = new_state
        return

    def dynamics(self, x0: VehicleState, u: VehicleCommands, mean: bool = True) -> VehicleState:
        """ Get rate of change of states for given control inputs """
        dx = x0.vx
        dr = dx * math.tan(x0.delta) / (2.0 * self.vg.l_half)
        dy = dr * self.vg.l_half
        th_eq = x0.theta + dr / 2.0 if mean else x0.theta
        costh = math.cos(th_eq)
        sinth = math.sin(th_eq)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        return VehicleState(x=xdot, y=ydot, theta=dr, vx=u.acc, delta=u.ddelta)

    def get_footprint(self) -> RectOBB:
        # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
        return RectOBB(self.vg.w_half, self.vg.l_half, self._state.theta, self._state.x, self._state.y)

    def get_vertices(self) -> np.ndarray:
        """
        This gets the car vertices in the global reference frame (RF)
        """
        l2g: SE2value = SE2_from_xytheta((self._state.x, self._state.y, self._state.theta))
        vertices = np.array([[self._state.x, -self._state.x, -self._state.x, self._state.x],
                             [self._state.y, self._state.y, -self._state.y, -self._state.y],
                             [1, 1, 1, 1]])
        vertices = l2g @ vertices
        vertices = vertices[:-1, :]  # Remove last row
        return vertices

    def get_shapely(self) -> Polygon:
        vertices = self.get_vertices()
        rectangle = Polygon([*vertices[:, 0], *vertices[:, 1], *vertices[:, 2], *vertices[:, 3]])
        return rectangle

    def get_xytheta_pose(self) -> SE2value:
        return SE2_from_xytheta([self._state.x, self._state.y, self._state.theta])

    def get_geometry(self) -> VehicleGeometry:
        return self.vg

    def get_vehicle_model(self):
        return self
