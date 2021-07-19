from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Mapping, MutableMapping

from dataclasses import dataclass

from crash.agent import Agent
from games import PlayerName, X, U

SimTime = Decimal


@dataclass
class SimParameters:
    dt: Decimal


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


@dataclass
class SimContext:
    models: Mapping[PlayerName, SimModel]
    players: Mapping[PlayerName, Agent]
    time: SimTime
    log: SimulationLog
    param: SimParameters
    sim_terminated: bool
    seed: int

    def __post_init__(self):
        assert all([player in self.models for player in self.players])
