from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.emergency.emergency_types import S


@dataclass
class BehaviorParams(BaseParams):
    pass


Obs = TypeVar('Obs')
Rel = TypeVar('Rel')


class Behavior(ABC, Generic[Obs, Rel, S]):
    @abstractmethod
    def update_observations(self, new_obs: Obs):
        pass

    @abstractmethod
    def is_there_anyone_to_yield_to(self, rel_behavior: Rel) -> bool:
        pass

    @abstractmethod
    def is_emergency_subroutine_needed(self, rel_behavior: Rel) -> bool:
        pass

    def get_situation(self, at: float) -> S:
        pass
