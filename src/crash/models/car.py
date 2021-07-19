from dataclasses import dataclass, replace
from decimal import Decimal

from typing import Tuple

import math
import numpy as np
from enum import IntEnum

from frozendict import frozendict
from scipy.integrate import solve_ivp

from crash.models.structures import COLOUR
from crash.simulator_structures import SimModel


@dataclass
class VehicleGeometry:
    """ Geometry parameters of the vehicle"""

    m: float
    """ Car Mass [kg] """
    w: float
    """ Half width of car [m] """
    l: float
    """ Half length of car - dist from CoG to each axle [m] """
    colour: COLOUR
    """ Car colour """

    @classmethod
    def default(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=1000.0, w=1.0, l=2.0, colour=(1, 1, 1))


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleActions:
    acc: float
    """ Acceleration [m/s2] """
    dst: float
    """ Steering rate [rad/s] """
    idx = frozendict({"acc": 0, "dst": 1})
    """ Dictionary to get correct values from numpy arrays"""

    def __add__(self, other: "VehicleActions") -> "VehicleActions":
        if type(other) == type(self):
            return replace(acc=self.acc + other.acc, dst=self.dst + other.dst)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleActions") -> "VehicleActions":
        return self + (other * -1.0)

    def __mul__(self, factor: float) -> "VehicleActions":
        return replace(acc=self.acc * factor, dst=self.dst * factor)

    __rmul__ = __mul__

    def __truediv__(self, factor: float) -> "VehicleActions":
        return self * (1 / factor)

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.acc,
            self.dst
        ])


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleState:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    theta: float
    """ CoG heading [rad] """
    v: float
    """ CoG longitudinal velocity [m/s] """
    st: float
    """ Steering angle [rad] """
    idx = frozendict({"x": 0, "y": 1, "th": 2, "v": 3, "st": 4})
    """ Dictionary to get correct values from numpy arrays"""

    def __add__(self, other: "VehicleState") -> "VehicleState":
        if type(other) == type(self):
            return replace(
                x=self.x + other.x,
                y=self.y + other.y,
                th=self.theta + other.theta,
                v=self.v + other.v,
                st=self.st + other.st
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleState") -> "VehicleState":
        return self + (other * -1.0)

    def __mul__(self, factor: float) -> "VehicleState":
        return replace(
            x=self.x * factor,
            y=self.y * factor,
            th=self.theta * factor,
            v=self.v * factor,
            st=self.st * factor,
        )

    __rmul__ = __mul__

    def __truediv__(self, factor: float) -> "VehicleState":
        return self * (1 / factor)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta,
            self.v,
            self.st
        ])


@dataclass
class CarParameters:
    v_max: float
    v_min: float
    """ Maximum and Minimum velocities [m/s] """
    st_max: float
    """ Maximum steering angle [rad] """


class CarModel(SimModel):
    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""
    state: VehicleState

    def update(self, action: VehicleActions, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def stateactions_from_array(y: np.ndarray) -> [VehicleState, VehicleActions]:
            n_states = len(self.state.idx)
            state = VehicleState(x=y[VehicleState.idx["x"]],
                                 y=y[VehicleState.idx["y"]],
                                 theta=y[VehicleState.idx["theta"]],
                                 v=y[VehicleState.idx["v"]],
                                 st=y[VehicleState.idx["st"]])
            actions = VehicleActions(acc=y[VehicleActions.idx["acc"] + n_states],
                                     dst=y[VehicleActions.idx["dst"] + n_states])
            return state, actions

        def dynamics(t, y):
            state0, actions = stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions, mean=False)
            du = np.zeros([len(VehicleActions.idx)])
            return np.concatenate([dx.as_ndarray(), du])

        state_np = self.state.as_ndarray()
        action_np = action.as_ndarray()
        y0 = np.concatenate([state_np, action_np])
        result = solve_ivp(fun=dynamics, t_span=(0.0, float(dt)), y0=y0)

        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")
        new_state, _ = stateactions_from_array(result.y)
        self.state = new_state
        return new_state

    def dynamics(self, x0: VehicleState, u: VehicleActions, mean: bool = True) -> VehicleState:
        """ Get rate of change of states for given control inputs """
        dx = x0.v
        dr = dx * math.tan(x0.st) / (2.0 * self.vg.l)
        dy = dr * self.vg.l
        th_eq = x0.theta + dr / 2.0 if mean else x0.theta
        costh = math.cos(th_eq)
        sinth = math.sin(th_eq)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        return VehicleState(x=xdot, y=ydot, theta=dr, v=u.acc, st=u.dst)
