from dataclasses import replace
from typing import Optional, Any
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from homotopies.mpc import MpcFullKinCont


class MpcAgent(Agent):
    def __init__(self, target_pos):
        self.mpc_controller = MpcFullKinCont(target_pos)
        self.target = target_pos
        self.my_name: PlayerName = None
        self.my_state: VehicleState = None
        self.plot_horizon = self.mpc_controller.params.n_horizon

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self.my_state = sim_obs.players[self.my_name].state
        obs_name = 'P2'
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
        # visualize mpc plan
        future_states = [self.my_state]
        timestamps_mpc = [0]
        for time_step in range(self.plot_horizon):
            future_states += [self.get_future_state(time_step+1)]
            timestamps_mpc += [time_step + 1]
        trajectory_mpc = (DgSampledSequence[VehicleState](timestamps_mpc, values=future_states), 'gold')

        # visualize constraints
        timestamps_cons = list(range(20))
        constrains_left_lb = []
        constrains_left_ub = []
        constrains_right_lb = []
        constrains_right_ub = []
        obstacle_state = np.array([self.mpc_controller.obstacle_obs.x,
                                   self.mpc_controller.obstacle_obs.y,
                                   self.mpc_controller.obstacle_obs.theta,
                                   self.mpc_controller.obstacle_obs.vx,
                                   self.mpc_controller.obstacle_obs.delta])
        for idx in range(20):
            s = idx/20*np.linalg.norm(self.target)
            e_constraints = self.mpc_controller.constraints_obs(s, obstacle_state)
            left_lb = self.mpc_controller.frame_rotation(e_constraints[0][0], s, np.pi/2-self.mpc_controller.target_direction)
            left_ub = self.mpc_controller.frame_rotation(e_constraints[0][1], s, np.pi/2-self.mpc_controller.target_direction)
            right_lb = self.mpc_controller.frame_rotation(e_constraints[1][0], s, np.pi/2-self.mpc_controller.target_direction)
            right_ub = self.mpc_controller.frame_rotation(e_constraints[1][1], s, np.pi/2-self.mpc_controller.target_direction)
            constrains_left_lb += [VehicleState(x=left_lb[0], y=left_lb[1], theta=0, vx=0, delta=0)]
            constrains_left_ub += [VehicleState(x=left_ub[0], y=left_ub[1], theta=0, vx=0, delta=0)]
            constrains_right_lb += [VehicleState(x=right_lb[0], y=right_lb[1], theta=0, vx=0, delta=0)]
            constrains_right_ub += [VehicleState(x=right_ub[0], y=right_ub[1], theta=0, vx=0, delta=0)]
        if self.mpc_controller.homotopy_class == 0:
            left_color = 'gold'
            right_color = 'blue'
        else:
            left_color = 'blue'
            right_color = 'gold'
        trajectory_left_lb = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_left_lb), left_color)
        trajectory_left_ub = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_left_ub), left_color)
        trajectory_right_lb = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_lb), right_color)
        trajectory_right_ub = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_ub), right_color)

        # visualize target region
        timestamps_target = list(range(5))
        target = self.mpc_controller.target
        tolerance = self.mpc_controller.target_tolerance
        target = [VehicleState(x=target[0]-tolerance, y=target[1]-tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0]-tolerance, y=target[1]+tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0]+tolerance, y=target[1]+tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0]+tolerance, y=target[1]-tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0]-tolerance, y=target[1]-tolerance, theta=0, delta=0, vx=0)]
        trajectory_target = (DgSampledSequence[VehicleState](timestamps_target, values=target), 'red')
        return [trajectory_mpc,
                trajectory_left_lb,
                trajectory_left_ub,
                trajectory_right_lb,
                trajectory_right_ub,
                trajectory_target]

    def get_future_state(self, time_step: int) -> VehicleState:
        x_t = np.array(self.mpc_controller.mpc.opt_x_num['_x', time_step, 0, 0]).squeeze()
        future_state = VehicleState(x=x_t[0], y=x_t[1], theta=x_t[2], vx=x_t[3], delta=x_t[4])
        return future_state
