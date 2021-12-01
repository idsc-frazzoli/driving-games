from time import perf_counter
from typing import Optional, Any
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState

from homotopies import logger
from homotopies.mpcc import MpccFullKinCont


class MpccAgent(Agent):
    def __init__(self, ref_path):
        self.controller = MpccFullKinCont(ref_path)
        self.ref_path = ref_path
        self.my_name: PlayerName = None
        self.my_state: VehicleState = None
        self.plot_horizon = self.controller.params.n_horizon

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self.my_state = sim_obs.players[self.my_name].state
        obs_name = 'P2'
        for key in sim_obs.players.keys():
            if not key == self.my_name:
                obs_name = key
        self.controller.obstacle_obs = sim_obs.players[obs_name].state
        self.controller.obstacle_obs_flag = True
        current_s, current_d = self.controller.frame_rotation(self.my_state.x, self.my_state.y,
                                                              self.controller.ref_direction)
        x0 = np.array([self.my_state.x,
                       self.my_state.y,
                       self.my_state.theta,
                       self.my_state.vx,
                       self.my_state.delta,
                       current_s]).reshape(-1, 1)
        # set initial guess for all decision variables using solution from last iteration
        self.controller.mpc.x0 = x0
        self.controller.set_initial_guess()
        tic = perf_counter()
        u0 = self.controller.mpc.make_step(x0)
        toc = perf_counter()
        logger.info("Mpc time:", make_step=toc - tic)
        #commands = VehicleCommands(acc=u0[0][0], ddelta=u0[1][0])
        u=[self.controller.mpc.opt_x_num['_u', 0, 0, 'a'].__float__(),
           self.controller.mpc.opt_x_num['_u', 0, 0, 'v_delta'].__float__()]
        commands = VehicleCommands(acc=u0[0][0], ddelta=u0[1][0])

        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        trajectories = []
        # visualize mpc plan
        trajectories += [self.visualize_mpc_plan()]

        timestamps_cons_num = 30
        # # visualize obstacle prediction
        # trajectories += [self.visualize_obs_pred()]
        #
        # # visualize constraints
        # obs_state = np.array([self.mpc_controller.obstacle_obs.x,
        #                       self.mpc_controller.obstacle_obs.y,
        #                       self.mpc_controller.obstacle_obs.theta,
        #                       self.mpc_controller.obstacle_obs.vx,
        #                       self.mpc_controller.obstacle_obs.delta])
        # trajectory_constraints = self.visualize_constraints(timestamps_cons_num, obs_state)
        # trajectories += [t for t in trajectory_constraints]

        # visualize target region
        trajectories += [self.visualize_ref()]

        trajectories += [t for t in self.visualize_all_constraints(timestamps_cons_num)]

        return trajectories

    def get_future_state(self, time_step: int) -> VehicleState:
        x_t = np.array(self.controller.mpc.opt_x_num['_x', time_step, 0, 0]).squeeze()
        future_state = VehicleState(x=x_t[0], y=x_t[1], theta=x_t[2], vx=x_t[3], delta=x_t[4])
        return future_state

    def visualize_mpc_plan(self):
        future_states = [self.my_state]
        timestamps_mpc = [0]
        for time_step in range(self.plot_horizon):
            future_states += [self.get_future_state(time_step + 1)]
            timestamps_mpc += [time_step + 1]
        trajectory_mpc = (DgSampledSequence[VehicleState](timestamps_mpc, values=future_states), 'gold')
        return trajectory_mpc

    def visualize_constraints(self, timestamps_cons_num, obstacle_state):
        timestamps_cons = list(range(timestamps_cons_num))
        constrains_left_lb = []
        constrains_left_ub = []
        constrains_right_lb = []
        constrains_right_ub = []
        for idx in range(timestamps_cons_num):
            s = idx / timestamps_cons_num * np.linalg.norm(self.ref_path[1])
            d_constraints = self.controller.constraints_obs(s, obstacle_state)
            left_lb = self.controller.frame_rotation(s, d_constraints[0][0], -self.controller.ref_direction)
            left_ub = self.controller.frame_rotation(s, d_constraints[0][1], -self.controller.ref_direction)
            right_lb = self.controller.frame_rotation(s, d_constraints[1][0], -self.controller.ref_direction)
            right_ub = self.controller.frame_rotation(s, d_constraints[1][1], -self.controller.ref_direction)
            constrains_left_lb += [VehicleState(x=left_lb[0], y=left_lb[1], theta=0, vx=0, delta=0)]
            constrains_left_ub += [VehicleState(x=left_ub[0], y=left_ub[1], theta=0, vx=0, delta=0)]
            constrains_right_lb += [VehicleState(x=right_lb[0], y=right_lb[1], theta=0, vx=0, delta=0)]
            constrains_right_ub += [VehicleState(x=right_ub[0], y=right_ub[1], theta=0, vx=0, delta=0)]
        if self.controller.homotopy_class == 0:
            left_color = 'gold'
            right_color = 'blue'
        else:
            left_color = 'blue'
            right_color = 'gold'
        trajectory_left_lb = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_left_lb), left_color)
        trajectory_left_ub = (DgSampledSequence[VehicleState](timestamps_cons, values=constrains_left_ub), left_color)
        trajectory_right_lb = (
            DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_lb), right_color)
        trajectory_right_ub = (
            DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_ub), right_color)

        return trajectory_left_lb, trajectory_left_ub, trajectory_right_lb, trajectory_right_ub

    def visualize_ref(self):
        ctr_pt_num = len(self.ref_path)
        timestamps_ref_path = list(range(ctr_pt_num))
        ref_path = []
        for ctr_pt in self.ref_path:
            ref_path += [VehicleState(x=ctr_pt[0], y=ctr_pt[1], theta=0, delta=0, vx=0)]
        trajectory_ref_path = (DgSampledSequence[VehicleState](timestamps_ref_path, values=ref_path), 'red')
        return trajectory_ref_path

    def visualize_obs_pred(self):
        obs_pred = []
        timestamps_obs_pred = list(range(self.controller.params.n_horizon + 1))
        obstacle_state = np.array([self.controller.obstacle_obs.x,
                                   self.controller.obstacle_obs.y,
                                   self.controller.obstacle_obs.theta,
                                   self.controller.obstacle_obs.vx,
                                   self.controller.obstacle_obs.delta])
        t_step = self.controller.params.t_step
        for k in range(self.controller.params.n_horizon + 1):
            obstacle_state[0] += obstacle_state[3] * t_step * np.cos(obstacle_state[2])
            obstacle_state[1] += obstacle_state[3] * t_step * np.sin(obstacle_state[2])
            obstacle_state[2] += t_step * obstacle_state[4]
            obs_pred += [VehicleState(x=obstacle_state[0], y=obstacle_state[1], theta=obstacle_state[2], vx=0, delta=0)]
        trajectory_obs_pred = (DgSampledSequence[VehicleState](timestamps_obs_pred, values=obs_pred), 'black')
        return trajectory_obs_pred

    def visualize_all_constraints(self, timestamps_cons_num):
        trajectories = []
        obs_pred = []
        timestamps_obs_pred = list(range(self.controller.params.n_horizon + 1))
        obstacle_state = np.array([self.controller.obstacle_obs.x,
                                   self.controller.obstacle_obs.y,
                                   self.controller.obstacle_obs.theta,
                                   self.controller.obstacle_obs.vx,
                                   self.controller.obstacle_obs.delta])
        t_step = self.controller.params.t_step
        for k in range(self.controller.params.n_horizon + 1):
            obstacle_state[0] += obstacle_state[3] * t_step * np.cos(obstacle_state[2])
            obstacle_state[1] += obstacle_state[3] * t_step * np.sin(obstacle_state[2])
            obstacle_state[2] += t_step * obstacle_state[4]
            obs_pred += [VehicleState(x=obstacle_state[0], y=obstacle_state[1], theta=obstacle_state[2], vx=0, delta=0)]
            trajectories += [t for t in self.visualize_constraints(timestamps_cons_num, obstacle_state)]
        trajectory_obs_pred = (DgSampledSequence[VehicleState](timestamps_obs_pred, values=obs_pred), 'black')
        trajectories += [trajectory_obs_pred]
        return trajectories
