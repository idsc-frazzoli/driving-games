from typing import Optional, Any

from dg_commons import PlayerName
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
from homotopies.mpc import MpcFullKinCont
#from homotopies.example_full_kin_mpc import FullMPCKin

class MpcAgent(Agent):
    def __init__(self, mpc_controller: MpcFullKinCont = MpcFullKinCont()):
        self.mpc_controller = mpc_controller
        self.my_name: str = ""

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        # todo
        pass

    def on_get_extra(
        self,
    ) -> Optional[Any]:
        # todo
        pass
