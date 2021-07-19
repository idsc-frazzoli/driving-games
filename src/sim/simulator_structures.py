from abc import ABC, abstractmethod
from decimal import Decimal
from typing import MutableMapping

from dataclasses import dataclass

from games import PlayerName, X, U

__all__ = ["SimTime", "SimObservations", "SimParameters", "SimModel", "SimulationLog", "LogEntry"]

SimTime = Decimal


@dataclass
class SimParameters:
    dt: Decimal
    """Simulation step in seconds"""


@dataclass
class SimObservations:
    players: MutableMapping[PlayerName, X]
    time: SimTime


@dataclass(unsafe_hash=True, frozen=True)
class LogEntry:
    state: X
    actions: U


SimulationLog = MutableMapping[SimTime, MutableMapping[PlayerName, LogEntry]]


class SimModel(ABC):
    state: X

    @abstractmethod
    def update(self, U, dt: SimTime) -> X:
        pass

    @abstractmethod
    def get_footprint(self):
        pass
