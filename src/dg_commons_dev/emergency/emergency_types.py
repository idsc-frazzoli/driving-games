from abc import abstractmethod
from typing import Optional
from dataclasses import dataclass
from duckietown_world import SE2Transform
from dg_commons.maps.lanes import DgLanelet
from sim_dev.agents import Agent
from dg_commons.sim import PlayerObservations
from dg_commons_dev.utils import BaseParams


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


class Emergency:
    @abstractmethod
    def update_emergency_situation(self, situation: EmergencySituation):
        pass

    @abstractmethod
    def get_new_path(self, path: DgLanelet) -> DgLanelet:
        pass

    @abstractmethod
    def get_new_speed_ref(self, speed_ref: float) -> float:
        pass
