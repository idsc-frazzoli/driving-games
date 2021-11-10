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
        self.target_direction = np.arctan2(target[0], target[1])
        self.target_tolerance = 2

        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.set_up_mpc()

    def set_up_mpc(self):
        """
        This method sets up the mpc and needs to be called in the inheriting __init__ method after the model setup """

        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)

        lterm = self.lterm(self.target[0], self.target[1], self.target_tolerance, 0)
        mterm = self.mterm(self.target[0], self.target[1], self.target_tolerance, 0)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )

        homotopy_class = np.array([1])
        self.mpc.set_uncertainty_values(homotopy=homotopy_class)

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
        #print('t_now ', t_now, 'curvilinear_s', self.model.aux['curvilinear_s'])
        if self.obstacle_obs_flag:
            obstacle_state = np.array([self.obstacle_obs.x,
                                       self.obstacle_obs.y,
                                       self.obstacle_obs.theta,
                                       self.obstacle_obs.vx,
                                       self.obstacle_obs.delta]).reshape(5, 1)
        else:
            obstacle_state = np.zeros([5, 1])
        for k in range(self.params.n_horizon+1):
            self.tvp_temp['_tvp', k, 'obstacle_state'] = obstacle_state
        return self.tvp_temp

    def lterm(self, target_x, target_y, target_tolerance, speed_ref, target_angle=None):
        error_x = if_else(fabs(target_x - self.state_x) < target_tolerance, 0, target_x - self.state_x)
        error_y = if_else(fabs(target_y - self.state_y) < target_tolerance, 0, target_y - self.state_y)
        error = [error_x, error_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        lterm, _ = self.cost.cost_function(error, inp)
        return lterm

    def mterm(self, target_x, target_y, target_tolerance, speed_ref, target_angle=None):
        error_x = if_else(fabs(target_x - self.state_x) < target_tolerance, 0, target_x - self.state_x)
        error_y = if_else(fabs(target_y - self.state_y) < target_tolerance, 0, target_y - self.state_y)
        error = [error_x, error_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        _, mterm = self.cost.cost_function(error, inp)
        return mterm

    def set_bounds(self):
        """ Here you might set bounds with time-varying parameters. """
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]
        state_s, state_e = self.world2curvilinear(self.state_x, self.state_y)
        self.mpc.set_nl_cons('lb_right', self.constraints_obs(state_s)[1][0] - state_e, ub=0, soft_constraint=True)
        self.mpc.set_nl_cons('ub_right', state_e - self.constraints_obs(state_s)[1][1], ub=0, soft_constraint=True)

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
        self.mpc.scaling['_u', 'a'] = 1

    def constraints_obs(self, s):
        obs_width_half = self.params.vehicle_geometry.w_half
        obs_lf = self.params.vehicle_geometry.lf
        obs_lr = self.params.vehicle_geometry.lr
        obs_s, obs_e = self.world2curvilinear(self.obstacle_state[0], self.obstacle_state[1])
        corner_left_rear = [obs_s - obs_lr, obs_e - obs_width_half]
        corner_left_front = [obs_s - obs_lr, obs_e - obs_width_half]
        corner_right_front = [obs_s + obs_lf, obs_e + obs_width_half]
        corner_right_rear = [obs_s + obs_lf, obs_e - obs_width_half]
        e_lb_r = -20
        e_lb_l = -20
        e_ub_r = 20
        e_ub_l = 20
        e_lb_r = if_else(s < corner_left_front[0], corner_left_front[1], e_lb_r)
        return [[e_lb_l, e_ub_l], [3, 10]]

    def world2curvilinear(self, x, y):
        s = x * cos(self.target_direction) + y * sin(self.target_direction)
        e = x * cos(self.target_direction-pi/2) + y * sin(self.target_direction-pi/2)
        return s, e
