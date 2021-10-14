from abc import ABC, abstractmethod
from sim_dev.agents import Agent
from typing import MutableMapping, Tuple, Dict
from dg_commons import PlayerName
from dg_commons.sim import PlayerObservations
from dataclasses import dataclass
from duckietown_world import SE2Transform
from dg_commons_dev.emergency.emergency_types import EmergencySituation
from dg_commons_dev.utils import BaseParams


@dataclass
class BehaviorParams(BaseParams):
    pass


class Behavior(ABC):
    @abstractmethod
    def update_observations(self, agents: MutableMapping[PlayerName, PlayerObservations]):
        pass

    @abstractmethod
    def is_there_anyone_to_yield_to(self, agents_rel_pose: Dict[PlayerName, SE2Transform]) -> bool:
        pass

    @abstractmethod
    def is_emergency_subroutine_needed(self, agents_rel_pose: Dict[PlayerName, SE2Transform]) -> bool:
        pass

    def get_situation(self, at: float) -> Tuple[float, EmergencySituation]:
        pass
