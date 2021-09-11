from games import PlayerName, U
from sim import SimObservations
from sim.agents.agent import Agent

__all__ = ["B2Agent"]



class B2Agent(Agent):
    """Baseline 2 agent"""

    def on_episode_init(self, my_name: PlayerName):
        pass

    def get_commands(self, sim_obs: SimObservations) -> U:
        pass

