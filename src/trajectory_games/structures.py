import os
from dataclasses import dataclass
from decimal import Decimal as D
from typing import FrozenSet, Tuple, Dict
from yaml import safe_load

from world import Lane, SE2Transform
from .config import config_dir

__all__ = [
    "VehicleGeometry",
    "VehicleActions",
    "VehicleState",
    "TrajectoryParams",
]


@dataclass
class VehicleGeometry:
    m: D
    """ Car Mass [kg] """
    w: D
    """ Car width [m] """
    l: D
    """ Half length of car - dist from CoG to each axle [m] """
    colour: Tuple[float, float, float]
    """ Car colour """

    _config: Dict = None
    """ Cached config, loaded from file """

    @classmethod
    def default(cls) -> "VehicleGeometry":
        return VehicleGeometry(m=D("1000"), w=D("1.0"), l=D("2.0"), colour=(1, 1, 1))

    @classmethod
    def load_colour(cls, name: str) -> Tuple[float, float, float]:
        def default():
            return 1, 1, 1

        if len(name) == 0:
            return default()
        if cls._config is None:
            filename = os.path.join(config_dir, "vehicles.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)
        if name in cls._config["colours"].keys():
            colour = tuple(_ for _ in cls._config["colours"][name])
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            colour = default()
        return colour

    @classmethod
    def from_config(cls, name: str) -> "VehicleGeometry":
        if len(name) == 0:
            return cls.default()
        if cls._config is None:
            filename = os.path.join(config_dir, "vehicles.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)
        if name in cls._config.keys():
            config = cls._config[name]
            vg = VehicleGeometry(
                m=D(config["m"]),
                w=D(config["w"]),
                l=D(config["l"]),
                colour=cls.load_colour(config["colour"]),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            vg = cls.default()
        return vg


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

    _config: Dict = None
    """ Cached config, loaded from file """

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

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

    @classmethod
    def default(cls, lane: Lane) -> "VehicleState":
        beta0 = lane.beta_from_along_lane(along_lane=0)
        se2 = SE2Transform.from_SE2(lane.center_point(beta=beta0))
        state = VehicleState(x=D(se2.p[0]), y=D(se2.p[1]), th=D(se2.theta), v=D("10"), st=D("0"), t=D("0"))
        return state

    @classmethod
    def from_config(cls, name: str, lane: Lane) -> "VehicleState":

        if len(name) == 0:
            return cls.default(lane)

        if cls._config is None:
            filename = os.path.join(config_dir, "initial_states.yaml")
            with open(filename) as load_file:
                cls._config = safe_load(load_file)
        if name in cls._config.keys():
            config = cls._config[name]
            beta0 = lane.beta_from_along_lane(along_lane=config["s0"])
            se2 = SE2Transform.from_SE2(lane.center_point(beta=beta0))
            state = VehicleState(
                x=D(se2.p[0]),
                y=D(se2.p[1]),
                th=D(se2.theta),
                v=D(config["v0"]),
                st=D(config["st0"]),
                t=D(config["t0"]),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            state = cls.default(lane)
        return state


@dataclass
class TrajectoryParams:
    max_gen: int
    dt: D
    u_acc: FrozenSet[D]
    u_dst: FrozenSet[D]
    v_max: D
    v_min: D
    st_max: D
    vg: VehicleGeometry

    _config: Dict = None
    """ Cached config, loaded from file """

    @classmethod
    def default(cls) -> "TrajectoryParams":
        u_acc = frozenset([D("-1"), D("0"), D("1")])
        u_dst = frozenset([_ * D("0.2") for _ in u_acc])
        params = TrajectoryParams(
            max_gen=1,
            dt=D("1"),
            u_acc=u_acc,
            u_dst=u_dst,
            v_max=D("15"),
            v_min=D("0"),
            st_max=D("0.5"),
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
            u_acc = frozenset(
                [
                    D(_ * config["step_acc"])
                    for _ in range(-config["n_acc"] // 2 + 1, config["n_acc"] // 2 + 1)
                ]
            )
            u_dst = frozenset(
                [
                    D(_ * config["step_dst"])
                    for _ in range(-config["n_dst"] // 2 + 1, config["n_dst"] // 2 + 1)
                ]
            )
            params = TrajectoryParams(
                max_gen=config["max_gen"],
                dt=D(config["dt"]),
                u_acc=u_acc,
                u_dst=u_dst,
                v_max=D(config["v_max"]),
                v_min=D(config["v_min"]),
                st_max=D(config["st_max"]),
                vg=VehicleGeometry.from_config(vg_name),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            params = cls.default()
        return params
