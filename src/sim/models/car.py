from dataclasses import dataclass, replace
from decimal import Decimal

import math
import numpy as np

from frozendict import frozendict
from scipy.integrate import solve_ivp

from sim.models.structures import Colour
from sim.simulator_structures import SimModel


@dataclass
class VehicleGeometry:
    """ Geometry parameters of the vehicle"""

    m: float
    """ Car Mass [kg] """
    w: float
    """ Half width of car [m] """
    l: float
    """ Half length of car - dist from CoG to each axle [m] """
    colour: Colour
    """ Car colour """

    @classmethod
    def default(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=1000.0, w=1.0, l=2.0, colour=(1, 1, 1))


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
            return replace(acc=self.acc + other.acc, dst=self.ddelta + other.ddelta)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleCommands") -> "VehicleCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "VehicleCommands":
        return replace(acc=self.acc * val, dst=self.ddelta * val)

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
            return replace(
                x=self.x + other.x,
                y=self.y + other.y,
                th=self.theta + other.theta,
                vx=self.vx + other.vx,
                delta=self.delta + other.delta
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleState") -> "VehicleState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "VehicleState":
        return replace(
            x=self.x * val,
            y=self.y * val,
            th=self.theta * val,
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


@dataclass
class CarParameters:
    vx_max: float
    vx_min: float
    """ Maximum and Minimum velocities [m/s] """
    delta_max: float
    """ Maximum steering angle [rad] """


class CarModel(SimModel):
    # fixme better to move it to object instance
    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""
    state: VehicleState

    def update(self, action: VehicleCommands, dt: Decimal):
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

        state_np = self.state.as_ndarray()
        action_np = action.as_ndarray()
        y0 = np.concatenate([state_np, action_np])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(dt)), y0=y0)

        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")
        new_state, _ = _stateactions_from_array(result.y)
        self.state = new_state
        return new_state

    def dynamics(self, x0: VehicleState, u: VehicleCommands, mean: bool = True) -> VehicleState:
        """ Get rate of change of states for given control inputs """
        dx = x0.vx
        dr = dx * math.tan(x0.delta) / (2.0 * self.vg.l)
        dy = dr * self.vg.l
        th_eq = x0.theta + dr / 2.0 if mean else x0.theta
        costh = math.cos(th_eq)
        sinth = math.sin(th_eq)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        return VehicleState(x=xdot, y=ydot, theta=dr, vx=u.acc, delta=u.ddelta)
