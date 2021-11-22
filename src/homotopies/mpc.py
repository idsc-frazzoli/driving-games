from dataclasses import dataclass

import numpy as np
from casadi import *

import do_mpc

__all__ = ["MpcFullKinCont"]

from homotopies.mpc_base import MpcKinBase, MpcKinBaseParams


class MpcFullKinCont(MpcKinBase):
    """ Nonlinear MPC contouring control of vehicle. Kinematic model without prior discretization """

    def __init__(self, target, params: MpcKinBaseParams = MpcKinBaseParams()):
        model_type = "continuous"  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        dtheta = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        vy = dtheta * self.params.vehicle_geometry.lr
        self.model.set_rhs("state_x", self.v * cos(self.theta) - vy * sin(self.theta))
        self.model.set_rhs("state_y", self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs("theta", dtheta)
        self.model.set_rhs("v", self.a)
        self.model.set_rhs("delta", self.v_delta)

        self.target = target
        self.target_direction = np.arctan2(target[1], target[0])
        self.target_tolerance = 3

        self.homotopy_class = np.array([0])  # 0 for overtaking from left, 1 for right

        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.set_up_mpc()

    def set_up_mpc(self):
        """
        This method sets up the mpc and needs to be called in the inheriting __init__ method after the model setup """

        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        lterm = self.lterm(self.target[0], self.target[1], self.target_tolerance, 5)
        mterm = self.mterm(self.target[0], self.target[1], self.target_tolerance, 5)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )

        self.mpc.set_uncertainty_values(homotopy=self.homotopy_class)

        self.set_bounds()
        self.set_scaling()

        self.tvp_temp = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_func)

        self.mpc.setup()

    def tvp_func(self, t_now):
        """
        Function describing the behavior of the time-varying variable.
        It has to take time as input argument and can be deterministic and non-deterministic (simple function of time
        or, as in this case, can change depending on incoming observations).
        """
        if self.obstacle_obs_flag:
            obstacle_state = np.array([self.obstacle_obs.x,
                                       self.obstacle_obs.y,
                                       self.obstacle_obs.theta,
                                       self.obstacle_obs.vx,
                                       self.obstacle_obs.delta]).reshape(5, 1)
        else:
            obstacle_state = np.zeros([5, 1])
        # predicted future position of the obstacle based on its current state
        for k in range(self.params.n_horizon + 1):
            obstacle_state[0] += obstacle_state[3] * self.params.t_step * cos(obstacle_state[2])
            obstacle_state[1] += obstacle_state[3] * self.params.t_step * sin(obstacle_state[2])
            obstacle_state[2] += self.params.t_step * obstacle_state[4]
            self.tvp_temp['_tvp', k, 'obstacle_state'] = obstacle_state
        return self.tvp_temp

    def lterm(self, target_x, target_y, target_tolerance, speed_ref, target_angle=None):
        error_x = if_else(fabs(target_x - self.state_x) < target_tolerance, 0, target_x - self.state_x)
        error_y = if_else(fabs(target_y - self.state_y) < target_tolerance, 0, target_y - self.state_y)
        error_speed = if_else(logic_or(error_x, error_y), self.v - speed_ref, self.v)
        error = [error_x, error_y, error_speed]
        inp = [self.v_delta, self.a]
        lterm, _ = self.cost.cost_function(error, inp)
        return lterm

    def mterm(self, target_x, target_y, target_tolerance, speed_ref, target_angle=None):
        error_x = if_else(fabs(target_x - self.state_x) < target_tolerance, 0, target_x - self.state_x)
        error_y = if_else(fabs(target_y - self.state_y) < target_tolerance, 0, target_y - self.state_y)
        error_speed = if_else(logic_or(error_x, error_y), self.v - speed_ref, self.v)
        error = [error_x, error_y, error_speed]
        inp = [self.v_delta, self.a]
        _, mterm = self.cost.cost_function(error, inp)
        return mterm

    def set_bounds(self):
        """ Here you might set bounds with time-varying parameters. """
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0] + 0.01
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1] - 0.01
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0] + 0.01
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1] - 0.01
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0] + 0.01
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1] - 0.01
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0] + 0.01
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1] - 0.01

        state_s, state_d = self.frame_rotation(self.state_x, self.state_y, self.target_direction)
        # self.mpc.set_nl_cons('lb_right',
        #                      self.homotopy * (self.constraints_obs(state_s, self.obstacle_state)[1][0] - state_d + 1.5),
        #                      ub=0)
        # self.mpc.set_nl_cons('ub_right',
        #                      self.homotopy * (state_d - self.constraints_obs(state_s, self.obstacle_state)[1][1] + 1.5),
        #                      ub=0)
        self.mpc.set_nl_cons('lb_left',
                             (1 - self.homotopy) * (
                                         self.constraints_obs(state_s, self.obstacle_state)[0][0] - state_d + 1.5),
                             ub=0)
        self.mpc.set_nl_cons('ub_left', (1 - self.homotopy) * (
                state_d - self.constraints_obs(state_s, self.obstacle_state)[0][1] + 1.5), ub=0)

    def constraints_obs(self, vehicle_s, obstacle_state):
        # get obstacle model parameters
        obs_w_half = self.params.vehicle_geometry.w_half
        obs_lf = self.params.vehicle_geometry.lf
        obs_lr = self.params.vehicle_geometry.lr
        # center coordinate of the obstacle in s-d frame
        obs_s, obs_d = self.frame_rotation(obstacle_state[0], obstacle_state[1], self.target_direction)
        # 4 corners' coordinates of the obstacle in s-d frame
        #   2---------1
        #   |    ->   |
        #   3---------4
        theta_diff = obstacle_state[2] - self.target_direction
        corner1_s, corner1_d = self.frame_rotation(2 * obs_lf, 2 * obs_w_half, -theta_diff)
        corner2_s, corner2_d = self.frame_rotation(-2 * obs_lr, 2 * obs_w_half, -theta_diff)
        corner3_s, corner3_d = self.frame_rotation(-2 * obs_lr, -2 * obs_w_half, -theta_diff)
        corner4_s, corner4_d = self.frame_rotation(2 * obs_lf, -2 * obs_w_half, -theta_diff)

        # case 1: theta_diff<pi/4, lane following scenario
        # left_front_s, left_front_d = corner4_s, corner4_d
        # left_rear_s, left_rear_d = corner1_s, corner1_d
        # right_rear_s, right_rear_d = corner2_s, corner2_d
        # right_front_s, right_front_d = corner3_s, corner3_d
        # case 2: theta_diff>pi/4, intersection scenario
        is_lane_following = if_else(fabs(theta_diff) < pi/4, 1, 0)
        heading_left = if_else(corner1_d-corner2_d > 0, 1, 0)  # heading=0: heading left, heading=1: heading right

        left_front_s = if_else(is_lane_following, corner1_s, if_else(heading_left, corner4_s, corner2_s))
        left_front_d = if_else(is_lane_following, corner1_d, if_else(heading_left, corner4_d, corner2_d))
        left_rear_s = if_else(is_lane_following, corner2_s, if_else(heading_left, corner1_s, corner3_s))
        left_rear_d = if_else(is_lane_following, corner2_d, if_else(heading_left, corner1_d, corner3_d))
        right_rear_s = if_else(is_lane_following, corner3_s, if_else(heading_left, corner2_s, corner4_s))
        right_rear_d = if_else(is_lane_following, corner3_d, if_else(heading_left, corner2_d, corner4_d))
        right_front_s = if_else(is_lane_following, corner4_s, if_else(heading_left, corner3_s, corner1_s))
        right_front_d = if_else(is_lane_following, corner4_d, if_else(heading_left, corner3_d, corner1_d))
        theta_diff = if_else(is_lane_following, theta_diff, if_else(heading_left, theta_diff-pi/2, theta_diff+pi/2))

        corner_left_rear = [obs_s + left_rear_s, obs_d + left_rear_d]
        corner_left_front = [obs_s + left_front_s, obs_d + left_front_d]
        corner_right_front = [obs_s + right_front_s, obs_d + right_front_d]
        corner_right_rear = [obs_s + right_rear_s, obs_d + right_rear_d]
        # get 4-line constraints
        safe_angle = pi / 3

        d_lb_r = -10
        d_ub_l = 10
        d_ub_r = if_else(vehicle_s+1 < corner_right_rear[0],
                         tan(fmax(-safe_angle + theta_diff, -pi/2+0.05)) * (vehicle_s - corner_right_rear[0]) + corner_right_rear[1],
                         if_else(vehicle_s < corner_right_front[0]+1,
                                 tan(theta_diff) * (vehicle_s - corner_right_rear[0]) + corner_right_rear[1],
                                 tan(fmin(safe_angle + theta_diff, pi/2-0.05)) * (vehicle_s - corner_right_front[0]) +
                                 corner_right_front[1]))
        d_lb_l = if_else(vehicle_s+1 < corner_left_rear[0],
                         tan(fmin(safe_angle + theta_diff, pi/2-0.05)) * (vehicle_s - corner_left_rear[0]) + corner_left_rear[1],
                         if_else(vehicle_s < corner_left_front[0]+1,
                                 tan(theta_diff) * (vehicle_s - corner_left_rear[0]) + corner_left_rear[1],
                                 tan(fmax(-safe_angle + theta_diff, -pi/2+0.05)) * (vehicle_s - corner_left_front[0]) + corner_left_front[
                                     1]))
        return [[d_lb_l, d_ub_l], [d_lb_r, d_ub_r]]

    def frame_rotation(self, x, y, theta):  # ref path: straight line from initial position(0, 0) to target position
        s = x * cos(theta) + y * sin(theta)
        d = -x * sin(theta) + y * cos(theta)
        return s, d

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
        self.mpc.scaling['_u', 'a'] = 1
