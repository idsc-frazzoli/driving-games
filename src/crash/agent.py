from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Callable

from crash.simulate import SimTime, SimObservations
from games import U, PlayerName, X


@dataclass
class AgentState:
    pass


@dataclass
class AgentAction:
    x: float
    y: str


class Agent(ABC):

    @abstractmethod
    def get_commands(self, sim_obs: SimObservations) -> U:
        pass


class NPAgent(Agent):
    """
    Non playing character which returns commands based purely on the sim time
    """

    def __init__(self, commands_plan: Mapping[SimTime, U]):
        self.commands_plan = commands_plan

    def get_commands(self, sim_obs: SimObservations) -> U:
        t: SimTime = sim_obs.time
        command = None
        for instant in self.commands_plan:
            if t > instant:
                command = self.commands_plan[instant]
            break
        if command is None:
            raise RuntimeError(f"Undefined command for instant {t}")
        return command


class PolicyAgent(Agent):
    """
    Non playing character which returns commands based purely on the sim time
    """

    def __init__(self, policy: Callable[[X], U], name: PlayerName):
        self.policy = policy
        self.my_name = name

    def get_commands(self, sim_obs: SimObservations) -> U:
        my_state: X = sim_obs.players[self.my_name]
        return self.policy(my_state)
