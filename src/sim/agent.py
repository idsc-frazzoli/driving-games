from abc import ABC, abstractmethod
from typing import Mapping, Callable, Union

from games import U, PlayerName, X
from sim import SimTime
from sim.simulator_structures import SimObservations

__all__ = ["Agent", "NPAgent", "PolicyAgent"]


class Agent(ABC):

    @abstractmethod
    def get_commands(self, sim_obs: SimObservations) -> U:
        pass


class NPAgent(Agent):
    """
    Non-playing character which returns commands based purely on the sim time
    This implements a ZOH policy based on the fact that the dictionaries are sorted since python 3.7
    """

    def __init__(self, commands_plan: Mapping[Union[SimTime, float], U]):
        self.commands_plan = commands_plan

    def get_commands(self, sim_obs: SimObservations) -> U:
        t: SimTime = sim_obs.time
        command = None
        for instant in self.commands_plan:
            if t >= instant:
                command = self.commands_plan[instant]
        if command is None:
            raise RuntimeError(f"Undefined command for instant {t}")
        return command


class PolicyAgent(Agent):
    """
    Playing character which returns commands based on its policy (function from state to commands)
    """

    def __init__(self, policy: Callable[[X], U], name: PlayerName):
        self.policy = policy
        self.my_name = name

    def get_commands(self, sim_obs: SimObservations) -> U:
        my_state: X = sim_obs.players[self.my_name]
        return self.policy(my_state)
