from typing import Optional, Any
import numpy as np
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
        current_state = sim_obs.players[self.my_name].state
        x0 = np.array([current_state.x,
                       current_state.y,
                       current_state.theta,
                       current_state.vx,
                       current_state.delta]).reshape(-1, 1)
        self.mpc_controller.mpc.x0 = x0
        self.mpc_controller.mpc.set_initial_guess()
        u0 = self.mpc_controller.mpc.make_step(x0)
        commands = VehicleCommands(acc=u0[0][0], ddelta=u0[1][0])
        return commands

    def on_get_extra(
        self,
    ) -> Optional[Any]:
        # todo
        pass
