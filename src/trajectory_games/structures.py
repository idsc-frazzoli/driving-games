import math
import os
from dataclasses import dataclass
from decimal import Decimal as D
from functools import cached_property
from typing import FrozenSet, Dict, List

import numpy as np
from geometry import SE2_from_xytheta
from typing import FrozenSet, Tuple, Dict

from yaml import safe_load

from dg_commons import Color
from .config import config_dir

__all__ = [
    "VehicleGeometry",
    "VehicleActions",
    "VehicleState",
    "TrajectoryParams",
]


# todo deprecated use things from dg_commons


@dataclass
class VehicleGeometry:
    """Geometry parameters of the vehicle"""

    m: float
    """ Car Mass [kg] """
    w: float
    """ Half width of car [m] """
    l: float
    """ Half length of car - dist from CoG to each axle [m] """
    colour: Color
    """ Car colour """

    _config: Dict = None
    """ Cached config, loaded from file """

    @classmethod
    def _load_all_configs(cls):
        if cls._config is None:
            filename = os.path.join(config_dir, "vehicles.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)

    @classmethod
    def default(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=1000.0, w=1.8, l=3.0, colour="k")

    @classmethod
    def load_colour(cls, name: str) -> Color:
        """Load the colour name from the possible colour configs"""

        if len(name) == 0:
            return "gray"
        return name

    @classmethod
    def from_config(cls, name: str) -> "VehicleGeometry":
        """Load the vehicle geometry from the possible vehicle configs"""
        if len(name) == 0:
            return cls.default()
        cls._load_all_configs()
        if name in cls._config.keys():
            config = cls._config[name]
            vg = VehicleGeometry(
                m=config["m"],
                w=config["w"],
                l=config["l"],
                colour=cls.load_colour(config["colour"]),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            vg = cls.default()
        return vg

    @cached_property
    def wheel_shape(self):
        halfwidth, radius = 0.1, 0.3  # size of the wheels
        return halfwidth, radius

    @cached_property
    def wheel_outline(self):
        halfwidth, radius = self.wheel_shape
        # fixme uniform points handlings to native list of tuples
        return np.array(
            [
                [radius, -radius, -radius, radius, radius],
                [-halfwidth, -halfwidth, halfwidth, halfwidth, -halfwidth],
                [1, 1, 1, 1, 1],
            ]
        )

    @cached_property
    def wheels_position(self) -> np.ndarray:
        halfwidth, radius = self.wheel_shape
        backwardshift = self.l / 4
        # return 4 wheels position (always the first half are the front ones)
        positions = np.array(
            [
                [self.l - radius - backwardshift, self.l - radius - backwardshift, -self.l + radius, -self.l + radius],
                [self.w - halfwidth, -self.w + halfwidth, self.w - halfwidth, -self.w + halfwidth],
                [1, 1, 1, 1],
            ]
        )
        return positions

    def get_rotated_wheels_outlines(self, delta: float) -> List[np.ndarray]:
        """
        :param delta: Steering angle of front wheels
        :return:
        """
        wheels_position = self.wheels_position
        transformed_wheels_outlines = []
        for i in range(4):
            # the first half of the wheels are the ones that get rotated
            if i < 2:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], delta))
            else:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], 0))
            transformed_wheels_outlines.append(transform @ self.wheel_outline)
        return transformed_wheels_outlines


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

    __radd__ = __add__

    def __sub__(self, other: "VehicleActions") -> "VehicleActions":
        return self + (other * -1.0)

    def __mul__(self, factor: float) -> "VehicleActions":
        return VehicleActions(acc=self.acc * factor, dst=self.dst * factor)

    __rmul__ = __mul__

    def __truediv__(self, factor: float) -> "VehicleActions":
        return self * (1 / factor)


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleState:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    th: float
    """ CoG heading [rad] """
    v: float
    """ CoG longitudinal velocity [m/s] """
    st: float
    """ Steering angle [rad] """
    t: D
    """ Time [s] """

    _config: Dict = None
    """ Cached config, loaded from file """

    @staticmethod
    def zero():
        return VehicleState(x=0.0, y=0.0, th=1.57, v=5.0, st=0.0, t=D("0"))

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

    def __sub__(self, other: "VehicleState") -> "VehicleState":
        return self + (other * -1.0)

    def __mul__(self, factor: float) -> "VehicleState":
        return VehicleState(
            x=self.x * factor,
            y=self.y * factor,
            th=self.th * factor,
            v=self.v * factor,
            st=self.st * factor,
            t=self.t * D(round(factor, 4)),
        )

    __rmul__ = __mul__

    def __truediv__(self, factor: float) -> "VehicleState":
        return self * (1 / factor)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

    def is_close(self, other: "VehicleState", tol: float = 1e-3) -> bool:
        diff = self - other

        def check(val: float) -> bool:
            return abs(val) < tol

        if (
            check(diff.x)
            and check(diff.y)
            and check(diff.th)
            and check(diff.v)
            and check(diff.th)
            and abs(diff.t) < 1e-3
        ):
            return True
        return False

    @classmethod
    def default(cls) -> "VehicleState":
        state = VehicleState(x=0.0, y=0.0, th=math.pi / 2.0, v=10.0, st=0.0, t=D("0"))
        return state

    @classmethod
    def from_config(cls, name: str) -> "VehicleState":

        if len(name) == 0:
            return cls.default()

        if cls._config is None:
            filename = os.path.join(config_dir, "initial_states.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)
        if name in cls._config.keys():
            config = cls._config[name]
            state = VehicleState(
                x=config["x0"],
                y=config["y0"],
                th=config["th0"],
                v=config["v0"],
                st=config["st0"],
                t=D(config["t0"]),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            state = cls.default()
        return state


@dataclass
class TrajectoryParams:
    solve: bool
    """ Generate trajectory by solving BVP at every stage or not """
    s_final: float
    """ Fraction of reference to generate trajectories - negative for finite time """
    max_gen: int
    """ Number of stages for trajectory generation """
    dt: D
    """ Sampling time [s] """

    u_acc: FrozenSet[float]
    u_dst: FrozenSet[float]
    """ Possible accelerations and steering rates to be sampled [m/s2] """

    v_max: float
    v_min: float
    """ Velocity hard limits [m/s] """

    st_max: float
    dst_max: float
    """ Steering angle and rate hard limits [rad],[rad/s] """

    dt_samp: D
    """ Timestamp for upsampling trajectories [s] """
    dst_scale: bool
    """ Scale target lateral deviation with velocity or not """
    n_factor: float
    """ Factor to scale target lateral deviation between stages """
    vg: VehicleGeometry
    """ Vehicle geometry parameters """

    _config: Dict = None
    """ Cached config, loaded from file """

    @classmethod
    def default(cls) -> "TrajectoryParams":
        u_acc = frozenset([-1.0, 0.0, 1.0])
        u_dst = frozenset([_ * 0.2 for _ in u_acc])
        params = TrajectoryParams(
            solve=False,
            s_final=-1.0,
            max_gen=1,
            dt=D("1"),
            u_acc=u_acc,
            u_dst=u_dst,
            v_max=15.0,
            v_min=0.0,
            st_max=0.5,
            dst_max=1.0,
            dt_samp=D("0.1"),
            dst_scale=False,
            n_factor=0.8,
            vg=VehicleGeometry.from_config(""),
        )
        return params

    @classmethod
    def from_config(cls, name: str, vg_name: str) -> "TrajectoryParams":
        if cls._config is None:
            filename = os.path.join(config_dir, "trajectories.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)

        if len(name) == 0:
            return cls.default()

        if name in cls._config.keys():
            config = cls._config[name]

            def get_set(inp: str):
                n = config["n_" + inp]
                start = -(n - 1) / 2.0
                u = frozenset([(_ + start) * config["step_" + inp] for _ in range(n)])
                return u

            u_acc = get_set(inp="acc")
            u_dst = get_set(inp="dst")
            params = TrajectoryParams(
                solve=config["solve"],
                s_final=config["s_final"],
                max_gen=config["max_gen"],
                dt=D(config["dt"]),
                u_acc=u_acc,
                u_dst=u_dst,
                v_max=config["v_max"],
                v_min=config["v_min"],
                st_max=config["st_max"],
                dst_max=config["dst_max"],
                dt_samp=D(config["dt_samp"]),
                dst_scale=config["dst_scale"],
                n_factor=config["n_factor"],
                vg=VehicleGeometry.from_config(vg_name),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            params = cls.default()
        return params
