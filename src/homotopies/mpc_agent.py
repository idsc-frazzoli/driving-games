from dataclasses import replace
from typing import Optional, Any
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from homotopies.mpc import MpcFullKinCont


class MpcAgent(Agent):
    def __init__(self, target_pos, mpc_controller: MpcFullKinCont = MpcFullKinCont()):
        self.mpc_controller = mpc_controller
        self.my_name: PlayerName = None
        self.my_state: VehicleState = None
        self.plot_horizon = self.mpc_controller.params.n_horizon

        self.mpc_controller.target = target_pos

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self.my_state = sim_obs.players[self.my_name].state
        for key in sim_obs.players.keys():
            if not key == self.my_name:
                obs_name = key
        self.mpc_controller.obstacle_obs = sim_obs.players[obs_name].state
        self.mpc_controller.obstacle_obs_flag = True
        x0 = np.array([self.my_state.x,
                       self.my_state.y,
                       self.my_state.theta,
                       self.my_state.vx,
                       self.my_state.delta]).reshape(-1, 1)
        self.mpc_controller.mpc.x0 = x0
        self.mpc_controller.mpc.set_initial_guess()
        u0 = self.mpc_controller.mpc.make_step(x0)
        commands = VehicleCommands(acc=u0[0][0], ddelta=u0[1][0])
        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        future_states = [self.my_state]
        timestamps = [0]
        for time_step in range(self.plot_horizon):
            future_states = future_states + [self.get_future_state(time_step+1)]
            timestamps = timestamps + [time_step+1]
        trajectory_mpc = DgSampledSequence[VehicleState](timestamps, values=future_states)
        return [(trajectory_mpc, "gold")]

    def get_future_state(self, time_step: int) -> VehicleState:
        x_t = np.array(self.mpc_controller.mpc.opt_x_num['_x', time_step, 0, 0]).squeeze()
        future_state = VehicleState(x=x_t[0], y=x_t[1], theta=x_t[2], vx=x_t[3], delta=x_t[4])
        return future_state
