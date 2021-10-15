from abc import abstractmethod, ABC
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass
from duckietown_world import SE2Transform
from dg_commons.maps.lanes import DgLanelet
from sim_dev.agents import Agent
from dg_commons.sim import PlayerObservations
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.controllers.interface import Ref
from dg_commons_dev.controllers.controller_types import Reference

S = TypeVar("S")


class Situation(ABC, Generic[S, Ref]):
    @abstractmethod
    def update_situation(self, situation: S):
        pass

    @abstractmethod
    def new_ref(self, current_ref: Ref) -> Ref:
        pass


@dataclass
class EmergencySituation:
    is_emergency: bool

    rel_pose: Optional[SE2Transform] = None
    my_agent: Optional[PlayerObservations] = None
    other_agent: Optional[PlayerObservations] = None
    other_vel: Optional[float] = None

    def __post_init__(self):
        if self.is_emergency:
            assert self.rel_pose is not None
            assert self.my_agent is not None
            assert self.other_agent is not None
            assert self.other_vel is not None


@dataclass
class EmergencyParams(BaseParams):
    pass


class Emergency(Situation[EmergencySituation, Reference]):
    @abstractmethod
    def update_situation(self, situation: EmergencySituation):
        pass

    @abstractmethod
    def get_new_path(self, path: DgLanelet) -> DgLanelet:
        pass

    @abstractmethod
    def get_new_speed_ref(self, speed_ref: float) -> float:
        pass

    def new_ref(self, current_ref: Reference) -> Reference:
        return Reference(path=self.get_new_path(current_ref.path),
                         speed_ref=self.get_new_speed_ref(current_ref.speed_ref))
