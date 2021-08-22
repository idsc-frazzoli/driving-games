from abc import ABC, abstractmethod
from typing import Callable, Optional

from dg_commons import DgSampledSequence
from games import U, PlayerName, X
from sim import SimTime
from sim.simulator_structures import SimObservations

__all__ = ["Agent", "NPAgent", "PolicyAgent"]


class Agent(ABC):

    @abstractmethod
    def on_episode_init(self, my_name: PlayerName):
        pass

    @abstractmethod
    def get_commands(self, sim_obs: SimObservations) -> U:
        pass


class NPAgent(Agent):
    """
    Non-playing character which returns commands based purely on the sim time
    """

    def __init__(self, commands_plan: DgSampledSequence[U]):
        self.commands_plan = commands_plan

    def on_episode_init(self, my_name: PlayerName):
        pass

    def get_commands(self, sim_obs: SimObservations) -> U:
        t: SimTime = sim_obs.time
        return self.commands_plan.at_or_previous(t)


class PolicyAgent(Agent):
    """
    Playing character which returns commands based on its policy (function from state to commands)
    """

    def __init__(self, policy: Callable[[X], U]):
        self.policy = policy
        self.my_name: Optional[PlayerName] = None

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> U:
        my_state: X = sim_obs.players[self.my_name]
        return self.policy(my_state)
