import os
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Dict, FrozenSet, Mapping, Optional, Union, List

from yaml import safe_load

from dg_commons import PlayerName, Timestamp
from dg_commons.planning import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.scenarios import DgScenario
from .config import CONFIG_DIR

__all__ = [
    "VehicleCommands",
    "TrajectoryGenParams",
]


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleCommands:
    acc: float
    """ Acceleration [m/s2] """
    dst: float
    """ Steering rate [rad/s] """

    def __add__(self, other: "VehicleCommands") -> "VehicleCommands":
        if type(other) == type(self):
            return VehicleCommands(acc=self.acc + other.acc, dst=self.dst + other.dst)
        elif other is None:
            return self
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "VehicleCommands") -> "VehicleCommands":
        return self + (other * -1.0)

    def __mul__(self, factor: float) -> "VehicleCommands":
        return VehicleCommands(acc=self.acc * factor, dst=self.dst * factor)

    __rmul__ = __mul__

    def __truediv__(self, factor: float) -> "VehicleCommands":
        return self * (1 / factor)


@dataclass
class TrajectoryGenParams:
    solve: bool
    """ Generate trajectory by solving BVP at every stage or not """
    s_final: float
    """ Fraction of reference to generate trajectories - negative for finite time """
    max_gen: int
    """ Number of stages for trajectory generation. """
    dt: D
    """ Sampling time [s] """

    u_acc: FrozenSet[float]
    u_dst: FrozenSet[float]
    """ Possible accelerations and steering rates [m/s2] """

    v_max: float
    v_min: float
    """ Velocity hard limits [m/s] """

    v_switch: float
    """Switching velocity of motor [m/s]"""

    acc_max: float
    """ Acceleration hard limits [m/s²]"""

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
    def default(cls) -> "TrajectoryGenParams":
        u_acc = frozenset([-1.0, 0.0, 1.0, 2.0])
        u_dst = frozenset([_ * 0.2 for _ in u_acc])
        params = TrajectoryGenParams(
            solve=False,
            s_final=-1.0,
            max_gen=1,
            dt=D("1"),
            u_acc=u_acc,
            u_dst=u_dst,
            v_max=15.0,
            v_min=0.0,
            v_switch=5.0,
            acc_max=10.0,
            st_max=0.5,
            dst_max=1.0,
            dt_samp=D("0.1"),
            dst_scale=False,
            n_factor=0.8,
            vg=VehicleGeometry.default_car(),
        )
        return params

    @classmethod
    def from_config(cls, name: str) -> "TrajectoryGenParams":
        if cls._config is None:
            filename = os.path.join(CONFIG_DIR, "trajectories.yaml")
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

            params = TrajectoryGenParams(
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
                vg=VehicleGeometry.default_car(),
            )
        else:
            print(f"Failed to initialise {cls.__name__} from {name}, using default")
            params = cls.default()
        return params


@dataclass
class TrajectoryGamePosetsParam:
    """
    This is a dataclass to store all relevant parameters when generating a Trajectory Game with players that have
    posetal preferences.

    Attributes:
        scenario:           DgScenario to use (builds on top of Commondroad scenario)
        initial_states:     Initial vehicle states for each player
        ref_lanes:          Reference lanes for each player
        pref_structures:    Name of preference structure used by each player (loaded from .yaml library file)
        traj_gen_params:    Parameters for the trajectory generator
        refresh_time:       Time for solving the game in a receding horizon fashion. In simulation, one refresh_time
                            has passed, the game will be prepreocessed and solved again with new observations.
                            Set to None if the game should be played only one, without Receding Horizon.
        n_traj_max:         Maximum number of trajectories for each player. These are sampled from the ones generated
                            by the trajectory generator. Can be an integer (if all players should have same number of
                            actions), or a mapping between PlayerName and an integer.
                            Set to None to keep all trajectories.
        sampling_method:    How to subsample trajectories. Can be "unif" or "uniform" for random uniform sampling, or
                            "variance" or "var" for looking for diverse and representative trajectories.
    """

    scenario: DgScenario
    initial_states: Mapping[PlayerName, VehicleState]
    ref_lanes: Mapping[PlayerName, List[RefLaneGoal]]
    pref_structures: Mapping[PlayerName, str]
    traj_gen_params: Mapping[PlayerName, TrajectoryGenParams]
    refresh_time: Optional[Timestamp] = None
    n_traj_max: Optional[Union[int, Mapping[PlayerName, int]]] = None
    sampling_method: str = "uniform"
