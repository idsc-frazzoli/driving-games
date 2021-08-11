from dataclasses import dataclass, replace
from functools import cached_property
from math import cos, sin
from typing import Tuple, Type, Sequence

import numpy as np
from frozendict import frozendict
from geometry import SE2value, T2value, SE2_from_xytheta
from scipy.integrate import solve_ivp
from shapely import affinity
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon

from sim import SimModel, SimTime
from sim.models.model_structures import ModelGeometry
from sim.models.model_utils import acceleration_constraint
from sim.models.pedestrian_utils import PedestrianParameters, rotation_constraint


@dataclass(frozen=True)
class PedestrianGeometry(ModelGeometry):

    @cached_property
    def outline(self) -> Sequence[Tuple[float, float]]:
        circle = Point(0, 0).buffer(1)  # type(circle)=polygon
        ellipse = affinity.scale(circle, 1, 1.5)  # not sure, maybe just a circle?
        return tuple(ellipse.exterior.coords)

    @classmethod
    def default(cls):
        return PedestrianGeometry(m=75, Iz=50, e=0.3, color="pink")


@dataclass(unsafe_hash=True, eq=True, order=True)
class PedestrianCommands:
    acc: float
    """ Acceleration [m/s^2] """
    dtheta: float
    """ rotational acceleration of the pedestrian"""
    idx = frozendict({"acc": 0, "dtheta": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "PedestrianCommands") -> "PedestrianCommands":
        if type(other) == type(self):
            return replace(self, acc=self.acc + other.acc, dtheta=self.dtheta + other.dtheta)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "PedestrianCommands") -> "PedestrianCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "PedestrianCommands":
        return replace(self, acc=self.acc * val, dtheta=self.dtheta * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "PedestrianCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.acc,
            self.dtheta
        ])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return PedestrianCommands(acc=z[cls.idx["acc"]], dtheta=z[cls.idx["dtheta"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class PedestrianState:
    x: float
    """ x-position of pedestrian [m] """
    y: float
    """ y-position of pedestrian [m] """
    theta: float
    """ orientation [rad] """
    vx: float
    """ longitudinal speed [m/s] """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "PedestrianState") -> "PedestrianState":
        if type(other) == type(self):
            return replace(self,
                           x=self.x + other.x,
                           y=self.y + other.y,
                           theta=self.theta + other.theta,
                           vx=self.vx + other.vx,
                           )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "PedestrianState") -> "PedestrianState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "PedestrianState":
        return replace(self,
                       x=self.x * val,
                       y=self.y * val,
                       theta=self.theta * val,
                       vx=self.vx * val,
                       )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "PedestrianState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta,
            self.vx,
        ])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return PedestrianState(x=z[cls.idx["x"]],
                               y=z[cls.idx["y"]],
                               theta=z[cls.idx["theta"]],
                               vx=z[cls.idx["vx"]],
                               )


class PedestrianModel(SimModel[SE2value, float]):

    def __init__(self, x0: PedestrianState, pg: PedestrianGeometry, pp: PedestrianParameters):
        self._state: PedestrianState = x0
        """ Current state of the model"""
        self.XT: Type[PedestrianState] = type(x0)
        """ State type"""
        self.pg: PedestrianGeometry = pg
        """ The vehicle's geometry parameters"""
        self.pp: PedestrianParameters = pp
        """ Pedestrian Parameters"""

    @classmethod
    def default(cls, x0: PedestrianState):
        return PedestrianModel(x0=x0, pg=PedestrianGeometry.default(), pp=PedestrianParameters.default())

    def update(self, commands: PedestrianCommands, dt: SimTime):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [PedestrianState, PedestrianCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            if self.has_collided:
                actions = PedestrianCommands(acc=0, dtheta=0)
            else:
                actions = PedestrianCommands(acc=y[PedestrianCommands.idx["acc"] + n_states],
                                             dtheta=y[PedestrianCommands.idx["dtheta"] + n_states])
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(PedestrianCommands.idx)])
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

    def dynamics(self, x0: PedestrianState, u: PedestrianCommands) -> PedestrianState:
        """ Simple double integrator"""
        # todo double integrator
        dtheta = rotation_constraint(rot_velocity=u.dtheta, pp=self.pp)
        acc = acceleration_constraint(speed=x0.vx, acceleration=u.acc, p=self.pp)
        return PedestrianState(
            x=x0.vx * cos(x0.theta),
            y=x0.vx * sin(x0.theta),
            theta=dtheta,
            vx=acc
        )

    def get_footprint(self) -> Polygon:
        footprint = Polygon(self.pg.outline)
        transform = self.get_pose()
        matrix_coeff = transform[0, :2].tolist() + transform[1, :2].tolist() + transform[:2, 2].tolist()
        footprint = affine_transform(footprint, matrix_coeff)
        assert footprint.is_valid
        return footprint

    def get_pose(self) -> SE2value:
        return SE2_from_xytheta(xytheta=(self._state.x, self._state.y, self._state.theta))

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        pass

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        pass

    def get_geometry(self) -> PedestrianGeometry:
        return self.pg
