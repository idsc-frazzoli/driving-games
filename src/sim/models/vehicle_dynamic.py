from dataclasses import dataclass, replace
from decimal import Decimal

import math

import numpy as np
from commonroad_dc.pycrcc import RectOBB

from frozendict import frozendict
from geometry import SE2value, SE2_from_xytheta
from scipy.integrate import solve_ivp

from sim.models.utils import kmh2ms
from sim.models.vehicle import VehicleCommands, VehicleGeometry, VehicleParameters
from sim.simulator_structures import SimModel


@dataclass(frozen=True, unsafe_hash=True)
class VehicleParametersDyn(VehicleParameters):
    """Provides additional parameters with respect to the kinematic model"""

    @classmethod
    def default_car(cls) -> "VehicleParametersDyn":
        return VehicleParametersDyn(vx_limits=(kmh2ms(-10), kmh2ms(130)), delta_max=math.pi)

    @classmethod
    def default_bicycle(cls) -> "VehicleParametersDyn":
        return VehicleParametersDyn(vx_limits=(kmh2ms(-1), kmh2ms(50)), delta_max=math.pi)

    def __post_init__(self):
        assert self.vx_limits[0] < self.vx_limits[1]


class VehicleTires:
    # todo
    pass


# Type alias since the commands are the same
VehicleCommandsDyn = VehicleCommands


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleStateDyn:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    theta: float
    """ CoG heading [rad] """
    vx: float
    """ CoG longitudinal velocity [m/s] """
    vy: float
    """ CoG longitudinal velocity [m/s] """
    dtheta: float
    """ yaw rate """
    delta: float
    """ Steering angle [rad] """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3, "vy": 4, "dtheta": 5, "delta": 6})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

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

    __radd__ = __add__

    def __sub__(self, other: "VehicleStateDyn") -> "VehicleStateDyn":
        return self + (other * -1.0)

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

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "VehicleStateDyn":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

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
                               dtheta=z[cls.idx["vx"]],
                               delta=z[cls.idx["delta"]])


class VehicleModelDyn(SimModel[VehicleStateDyn, VehicleCommands]):

    def __init__(self, x0: VehicleStateDyn, vg: VehicleGeometry, vp: VehicleParametersDyn, vtires: VehicleTires):
        self._state: VehicleStateDyn = x0
        """ Current state of the model"""
        self.vg: VehicleGeometry = vg
        """ The vehicle's geometry parameters"""
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

    def update(self, commands: VehicleCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [VehicleStateDyn, VehicleCommands]:
            n_states = VehicleStateDyn.get_n_states()
            state = VehicleStateDyn.from_array(y[0:n_states])
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

    def dynamics(self, x0: VehicleStateDyn, u: VehicleCommands, mean: bool = True) -> VehicleStateDyn:
        """ returns state derivative for given control inputs """

        # todo

        return VehicleStateDyn(x=xdot, y=ydot, theta=dtheta, vx=u.acc, delta=u.ddelta)

    def get_footprint(self) -> RectOBB:
        # Oriented rectangle with width/2, height/2, orientation, x-position , y-position
        return RectOBB(self.vg.w_half, self.vg.l_half, self._state.theta, self._state.x, self._state.y)

    def get_xytheta_pose(self) -> SE2value:
        return SE2_from_xytheta([self._state.x, self._state.y, self._state.theta])

    def get_geometry(self) -> VehicleGeometry:
        return self.vg
