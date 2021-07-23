from abc import ABC, abstractmethod
from copy import deepcopy
from decimal import Decimal
from typing import MutableMapping, Generic

from dataclasses import dataclass

from commonroad_dc.pycrcc import Shape
from geometry import SE2value

from games import PlayerName, X, U

__all__ = ["SimTime", "SimObservations", "SimParameters", "SimModel", "SimulationLog", "LogEntry"]

SimTime = Decimal


@dataclass(frozen=True, unsafe_hash=True)
class SimParameters:
    dt: Decimal = Decimal(0.2)
    """Simulation step in seconds"""
    max_sim_time: Decimal = Decimal(10)
    """Max Simulation time in seconds"""


@dataclass
class SimObservations:
    players: MutableMapping[PlayerName, X]
    time: SimTime


@dataclass(unsafe_hash=True, frozen=True)
class LogEntry:
    state: X
    actions: U


SimulationLog = MutableMapping[SimTime, MutableMapping[PlayerName, LogEntry]]


class SimModel(ABC, Generic[X, U]):
    _state: X

    @abstractmethod
    def update(self, commands: U, dt: SimTime) -> X:
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
