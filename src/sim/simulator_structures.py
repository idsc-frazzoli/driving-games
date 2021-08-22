from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import MutableMapping, Generic, Optional, Any, Dict, Union, Type, Mapping

from geometry import SE2value, T2value
from shapely.geometry import Polygon

from dg_commons.time import time_function
from games import PlayerName, X, U
from sim import SimTime, ImpactLocation

__all__ = ["SimObservations", "SimParameters", "SimModel", "SimulationLog", "LogEntry"]


@dataclass(frozen=True, unsafe_hash=True)
class SimParameters:
    dt: SimTime = SimTime(0.05)
    """Simulation step [s]"""
    dt_commands: SimTime = SimTime(0.1)
    """How often shall we ask the agents for new commands"""
    max_sim_time: SimTime = SimTime(6)
    """Max Simulation time overall [s]"""
    sim_time_after_collision: SimTime = SimTime(0)
    """The simulation time for which to continue after the first collision is detected [s]"""

    @classmethod
    def default(cls) -> "SimParameters":
        return SimParameters(dt=SimTime(0.05), max_sim_time=SimTime(10), sim_time_after_collision=SimTime(0))


@dataclass
class SimObservations:
    """The observations from the simulator passed to each agent"""
    players: MutableMapping[PlayerName, X]
    time: SimTime


@dataclass(unsafe_hash=True, frozen=True)
class LogEntry:
    state: X
    actions: U
    extra: Optional[Any] = None


class SimulationLog(Dict[SimTime, MutableMapping[PlayerName, LogEntry]]):
    # todo consider switching to DgSampledSequence

    def get_init_time(self) -> SimTime:
        return next(iter(self))

    def get_last_time(self) -> SimTime:
        return next(reversed(self))

    def get_entry_before(self, t: SimTime) -> (SimTime, MutableMapping[PlayerName, LogEntry]):
        sim_time = self.get_last_time()
        times = reversed(self)
        while sim_time > t:
            sim_time = next(times)
        return sim_time, self[sim_time]

    def get_entry_after(self, t: SimTime) -> (SimTime, MutableMapping[PlayerName, LogEntry]):
        sim_time = self.get_init_time()
        times = iter(self)
        while sim_time < t:
            sim_time = next(times)
        return sim_time, self[sim_time]

    def at(self, t: Union[SimTime, float]) -> MutableMapping[PlayerName, LogEntry]:
        t = Decimal(t)
        if t < self.get_init_time() or t > self.get_last_time():
            raise ValueError(f"Requested simulation log {t} is out of bounds")
        t0, entry_before = self.get_entry_before(t)
        t1, entry_after = self.get_entry_after(t)
        alpha: float = float(t - t0) / float(t1 - t0) if float(t1 - t0) > 0 else 0.0
        interpolated_entry: Dict[PlayerName, LogEntry] = {}
        for player in entry_before:
            interpolated_entry[player] = LogEntry(
                state=alpha * entry_after[player].state + (1 - alpha) * entry_before[player].state,
                actions=entry_before[player].actions,
                extra=entry_before[player].extra)

        return interpolated_entry


class SimModel(ABC, Generic[X, U]):
    _state: X
    """State of the model"""
    XT: Type[X] = object
    """Type of the state"""
    has_collided: bool = False
    """Whether or not the object has already collided"""

    @abstractmethod
    def update(self, commands: U, dt: SimTime):
        """ The model gets updated via this function """
        pass

    @abstractmethod
    def get_footprint(self) -> Polygon:
        """ This returns the footprint of the model that is used for collision checking"""
        pass

    @abstractmethod
    def get_pose(self) -> SE2value:
        """Return pose of the model"""
        pass

    @abstractmethod
    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Get velocity of the model, default in body frame, otherwise in global"""
        pass

    @abstractmethod
    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        """Set velocity of the model
        :param vel:
        :param omega:
        :param in_model_frame: If the passed value are already in body frame (True) or global (False)
        """
        pass

    @abstractmethod
    def get_geometry(self) -> Any:
        pass

    @abstractmethod
    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        pass

    def get_state(self) -> X:
        return deepcopy(self._state)
