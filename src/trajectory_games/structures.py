import os
from dataclasses import dataclass
from decimal import Decimal as D
from typing import FrozenSet, Tuple, Dict
from yaml import safe_load

from _tmp._deprecated.world import SE2Transform, LaneSegmentHashable
from .config import config_dir

__all__ = [
    "VehicleGeometry",
    "VehicleActions",
    "VehicleState",
    "TrajectoryParams",
]


@dataclass
class VehicleGeometry:
    """ Geometry parameters of the vehicle"""

    COLOUR = Tuple[float, float, float]
    """ An alias to store the RGB values of a colour """

    m: float
    """ Car Mass [kg] """
    w: float
    """ Half width of car [m] """
    l: float
    """ Half length of car - dist from CoG to each axle [m] """
    colour: COLOUR
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
        return VehicleGeometry(m=1000.0, w=1.0, l=2.0, colour=(1, 1, 1))

    @classmethod
    def load_colour(cls, name: str) -> COLOUR:
        """ Load the colour name from the possible colour configs"""

        def default():
            return 1, 1, 1

        if len(name) == 0:
            return default()
        cls._load_all_configs()
        if name in cls._config["colours"].keys():
            colour = tuple(_ for _ in cls._config["colours"][name])
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            colour = default()
        return colour

    @classmethod
    def from_config(cls, name: str) -> "VehicleGeometry":
        """ Load the vehicle geometry from the possible vehicle configs"""
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
        return self * (1/factor)


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
        return self * (1/factor)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("_")})

    def is_close(self, other: "VehicleState", tol: float = 1e-3) -> bool:
        diff = self - other

        def check(val: float) -> bool:
            return abs(val) < tol

        if check(diff.x) and check(diff.y) and check(diff.th) and \
                check(diff.v) and check(diff.th) and abs(diff.t) < 1e-3:
            return True
        return False

    @classmethod
    def default(cls, lane: LaneSegmentHashable) -> "VehicleState":
        beta0 = lane.beta_from_along_lane(along_lane=0)
        se2 = SE2Transform.from_SE2(lane.center_point(beta=beta0))
        state = VehicleState(x=se2.p[0], y=se2.p[1], th=se2.theta,
                             v=10.0, st=0.0, t=D("0"))
        return state

    @classmethod
    def from_config(cls, name: str, lane: LaneSegmentHashable) -> "VehicleState":

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
                x=se2.p[0],
                y=se2.p[1],
                th=se2.theta+config["th0"],
                v=config["v0"],
                st=config["st0"],
                t=D(config["t0"]),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            state = cls.default(lane)
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
                n = config["n_"+inp]
                start = -(n-1)/2.0
                u = frozenset([(_+start) * config["step_"+inp] for _ in range(n)])
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
                vg=VehicleGeometry.from_config(vg_name),
            )
        else:
            print(f"Failed to intialise {cls.__name__} from {name}, using default")
            params = cls.default()
        return params
