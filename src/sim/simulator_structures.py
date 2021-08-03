from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import MutableMapping, Generic, Optional, Any, Dict, Union

from commonroad_dc.pycrcc import Shape
from geometry import SE2value

from games import PlayerName, X, U

__all__ = ["SimTime", "SimObservations", "SimParameters", "SimModel", "SimulationLog", "LogEntry"]

SimTime = Decimal


@dataclass(frozen=True, unsafe_hash=True)
class SimParameters:
    dt: Decimal = Decimal(0.2)
    """Simulation step in seconds"""
    max_sim_time: Decimal = Decimal(6)
    """Max Simulation time in seconds"""


@dataclass
class SimObservations:
    players: MutableMapping[PlayerName, X]
    time: SimTime


@dataclass(unsafe_hash=True, frozen=True)
class LogEntry:
    state: X
    actions: U
    extra: Optional[Any] = None


class SimulationLog(Dict[SimTime, MutableMapping[PlayerName, LogEntry]]):
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
        # todo enhancement bisect_right and used SampledSequence
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

    @abstractmethod
    def update(self, commands: U, dt: SimTime):
        pass

    @abstractmethod
    def get_footprint(self) -> Shape:
        """ This returns the footprint of the model that is used for collision checking"""
        pass

    @abstractmethod
    def get_xytheta_pose(self) -> SE2value:
        pass

    def get_state(self) -> X:
        return deepcopy(self._state)

    @abstractmethod
    def get_geometry(self) -> Any:
        pass

    @abstractmethod
    def get_velocity(self) -> Any:
        pass
