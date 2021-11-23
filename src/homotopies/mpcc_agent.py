from dataclasses import replace
from typing import Optional, Any
import numpy as np
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from homotopies.mpc import MpccFullKinCont


class MpccAgent(Agent):
    def __init__(self, ref_path):
        self.mpc_controller = MpccFullKinCont(ref_path)
        self.ref_path = ref_path
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
        current_s, current_d = self.mpc_controller.frame_rotation(self.my_state.x, self.my_state.y, self.mpc_controller.ref_direction)
        x0 = np.array([self.my_state.x,
                       self.my_state.y,
                       self.my_state.theta,
                       self.my_state.vx,
                       self.my_state.delta,
                       current_s]).reshape(-1, 1)
        self.mpc_controller.mpc.x0 = x0
        self.mpc_controller.mpc.set_initial_guess()
        u0 = self.mpc_controller.mpc.make_step(x0)
        commands = VehicleCommands(acc=u0[0][0], ddelta=u0[1][0])
        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        # visualize mpc plan
        trajectory_mpc = self.visualize_mpc_plan()

        # visualize constraints
        timestamps_cons_num = 30
        trajectory_constraints = self.visualize_constraints(timestamps_cons_num)
        trajectory_left_lb, trajectory_left_ub, trajectory_right_lb, trajectory_right_ub = trajectory_constraints

        # visualize target region
        trajectory_target = self.visualize_target()

        # visualize predicted obstacle pos
        #trajectory_obstacle = self.visualize_obstacle()

        return [
            trajectory_mpc,
            trajectory_left_lb, trajectory_left_ub, trajectory_right_lb, trajectory_right_ub,
            trajectory_target,
            #trajectory_obstacle
        ]

    def get_future_state(self, time_step: int) -> VehicleState:
        x_t = np.array(self.mpc_controller.mpc.opt_x_num['_x', time_step, 0, 0]).squeeze()
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

    def visualize_constraints(self, timestamps_cons_num):
        timestamps_cons = list(range(timestamps_cons_num))
        constrains_left_lb = []
        constrains_left_ub = []
        constrains_right_lb = []
        constrains_right_ub = []
        obstacle_state = np.array([self.mpc_controller.obstacle_obs.x,
                                   self.mpc_controller.obstacle_obs.y,
                                   self.mpc_controller.obstacle_obs.theta,
                                   self.mpc_controller.obstacle_obs.vx,
                                   self.mpc_controller.obstacle_obs.delta])
        for idx in range(timestamps_cons_num):
            s = idx / timestamps_cons_num * np.linalg.norm(self.ref_path[0])
            d_constraints = self.mpc_controller.constraints_obs(s, obstacle_state)
            left_lb = self.mpc_controller.frame_rotation(s, d_constraints[0][0], -self.mpc_controller.ref_direction)
            left_ub = self.mpc_controller.frame_rotation(s, d_constraints[0][1], -self.mpc_controller.ref_direction)
            right_lb = self.mpc_controller.frame_rotation(s, d_constraints[1][0], -self.mpc_controller.ref_direction)
            right_ub = self.mpc_controller.frame_rotation(s, d_constraints[1][1], -self.mpc_controller.ref_direction)
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
        trajectory_right_lb = (
            DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_lb), right_color)
        trajectory_right_ub = (
            DgSampledSequence[VehicleState](timestamps_cons, values=constrains_right_ub), right_color)

        return trajectory_left_lb, trajectory_left_ub, trajectory_right_lb, trajectory_right_ub

    def visualize_target(self):
        timestamps_target = list(range(5))
        target = self.mpc_controller.target
        tolerance = 1
        target = [VehicleState(x=target[0] - tolerance, y=target[1] - tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0] - tolerance, y=target[1] + tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0] + tolerance, y=target[1] + tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0] + tolerance, y=target[1] - tolerance, theta=0, delta=0, vx=0),
                  VehicleState(x=target[0] - tolerance, y=target[1] - tolerance, theta=0, delta=0, vx=0)]
        trajectory_target = (DgSampledSequence[VehicleState](timestamps_target, values=target), 'red')
        return trajectory_target

    def visualize_obstacle(self):
        timestamps_target = list(range(5))
        obs_w_half = self.mpc_controller.params.vehicle_geometry.w_half
        obs_lf = self.mpc_controller.params.vehicle_geometry.lf
        obs_lr = self.mpc_controller.params.vehicle_geometry.lr
        obstacle_state = np.array([self.mpc_controller.obstacle_obs.x,
                                   self.mpc_controller.obstacle_obs.y,
                                   self.mpc_controller.obstacle_obs.theta,
                                   self.mpc_controller.obstacle_obs.vx,
                                   self.mpc_controller.obstacle_obs.delta])

        obs_s, obs_d = self.mpc_controller.frame_rotation(obstacle_state[0], obstacle_state[1],
                                                          self.mpc_controller.ref_direction)
        theta_diff = obstacle_state[2] - self.mpc_controller.ref_direction
        corner1_s, corner1_d = self.mpc_controller.frame_rotation(2 * obs_lf, 2 * obs_w_half, -theta_diff)
        corner2_s, corner2_d = self.mpc_controller.frame_rotation(-2 * obs_lr, 2 * obs_w_half, -theta_diff)
        corner3_s, corner3_d = self.mpc_controller.frame_rotation(-2 * obs_lr, -2 * obs_w_half, -theta_diff)
        corner4_s, corner4_d = self.mpc_controller.frame_rotation(2 * obs_lf, -2 * obs_w_half, -theta_diff)

        if abs(theta_diff) < np.pi / 4:
            left_front_s = corner1_s
            left_front_d = corner1_d
            left_rear_s = corner2_s
            left_rear_d = corner2_d
            right_rear_s = corner3_s
            right_rear_d = corner3_d
            right_front_s = corner4_s
            right_front_d = corner4_d
        elif corner1_d - corner2_d > 0:
            left_front_s = corner4_s
            left_front_d = corner4_d
            left_rear_s = corner1_s
            left_rear_d = corner1_d
            right_rear_s = corner2_s
            right_rear_d = corner2_d
            right_front_s = corner3_s
            right_front_d = corner3_d
        else:
            left_front_s = corner2_s
            left_front_d = corner2_d
            left_rear_s = corner3_s
            left_rear_d = corner3_d
            right_rear_s = corner4_s
            right_rear_d = corner4_d
            right_front_s = corner1_s
            right_front_d = corner1_d

        corner_left_rear = self.mpc_controller.frame_rotation(obs_s + left_rear_s, obs_d + left_rear_d,
                                                              -self.mpc_controller.ref_direction)
        corner_left_front = self.mpc_controller.frame_rotation(obs_s + left_front_s, obs_d + left_front_d,
                                                               -self.mpc_controller.ref_direction)
        corner_right_front = self.mpc_controller.frame_rotation(obs_s + right_front_s, obs_d + right_front_d,
                                                                -self.mpc_controller.ref_direction)
        corner_right_rear = self.mpc_controller.frame_rotation(obs_s + right_rear_s, obs_d + right_rear_d,
                                                               -self.mpc_controller.ref_direction)
        obstacle = [VehicleState(x=corner_left_rear[0], y=corner_left_rear[1], theta=0, delta=0, vx=0),
                    VehicleState(x=corner_left_front[0], y=corner_left_front[1], theta=0, delta=0, vx=0),
                    VehicleState(x=corner_right_front[0], y=corner_right_front[1], theta=0, delta=0, vx=0),
                    VehicleState(x=corner_right_rear[0], y=corner_right_rear[1], theta=0, delta=0, vx=0),
                    VehicleState(x=corner_right_rear[0]+0.5, y=corner_right_rear[1]+0.5, theta=0, delta=0, vx=0)
                    ]
        trajectory_obstacle = (DgSampledSequence[VehicleState](timestamps_target, values=obstacle), 'red')

        return trajectory_obstacle
