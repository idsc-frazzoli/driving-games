from time import perf_counter
from typing import Optional, Any
import numpy as np
import sys
import forcespro
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState

from homotopies import logger
from homotopies.MIP.MIP_solver import generate_pathplanner, create_model, set_bounds, MIPModelParams, get_obs_pred


class MIPAgent(Agent):
    def __init__(self, ref_path):
        self.ref_path = ref_path
        self.my_name: PlayerName = None
        self.my_state: VehicleState = None
        self.params = MIPModelParams
        self.solver = None
        self.model = None
        self.last_output = {}

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.model, self.solver = generate_pathplanner()
        # self.model = create_model()
        # self.solver = forcespro.nlp.Solver.from_directory("./FORCESNLPsolver")

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        self.my_state = sim_obs.players[self.my_name].state
        obs_name = 'P2'
        for key in sim_obs.players.keys():
            if not key == self.my_name:
                obs_name = key
        self.obstacle_obs = sim_obs.players[obs_name].state
        self.obstacle_obs_flag = True

        xinit = np.array([self.my_state.x,
                          self.my_state.y,
                          self.my_state.theta,
                          self.my_state.vx,
                          self.my_state.delta]).reshape(-1, 1)
        ub0 = np.array([1,0,0,0]).reshape(-1, 1)
        uc0 = np.zeros([self.params.nc, 1])

        z0 = np.concatenate([ub0, uc0, xinit])

        if not self.last_output:
            x0 = np.transpose(np.tile(z0, (1, self.model.N)))
        else:
            x0 = np.concatenate([self.last_output["ub"], self.last_output["uc"], self.last_output["x"]], axis=1)  # todo: check
            x0 = x0.flatten()

        # set initial guess and initial state
        problem = {"x0": x0,
                   "xinit": xinit}

        # Set runtime parameters, should be a single vector with length npar*N
        target_param = np.tile(self.ref_path[1], (self.model.N, 1))
        pred_param = get_obs_pred(self.obstacle_obs)
        all_param = np.concatenate([pred_param, target_param], axis=1)
        problem["all_parameters"] = all_param.flatten()

        # set bounds
        continuous_bounds = set_bounds()  # bounds for continuous variables
        problem["lb{:02d}".format(1)] = np.concatenate([0*np.ones(self.params.nb), continuous_bounds[:self.params.nc, 0]])
        problem["ub{:02d}".format(1)] = np.concatenate([1*np.ones(self.params.nb), continuous_bounds[:self.params.nc, 1]])
        for s in range(1, self.model.N):
            problem["lb{:02d}".format(s + 1)] = np.concatenate([0*np.ones(self.params.nb), continuous_bounds[:, 0]])
            problem["ub{:02d}".format(s + 1)] = np.concatenate([1*np.ones(self.params.nb), continuous_bounds[:, 1]])

        # call the solver
        output, exitflag, info = self.solver.solve(problem)
        # Make sure the solver has exited properly.

        assert exitflag == 1, "Optimization failed with exitflag:{}.\n".format(exitflag)
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
                         .format(info.it, info.solvetime))

        # Extract output
        # temp = np.zeros((np.max(self.model.nvar), self.model.N))
        # for i in range(0, self.model.N):
        #     temp[:, i] = output['x{0:02d}'.format(i + 1)]
        # self.last_output["u"] = temp[0:self.params.nu, :]
        # self.last_output["x"] = temp[self.params.nu:self.params.nz, :]
        self.last_output = output
        commands = VehicleCommands(acc=self.last_output["uc"][0][0],
                                   ddelta=self.last_output["uc"][1][0])

        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        trajectories = []
        # visualize control plan
        trajectories += [self.visualize_MIP_plan()]
        #
        # timestamps_cons_num = 30
        # # visualize obstacle prediction
        # # trajectories += [self.visualize_obs_pred()]
        #
        # # visualize constraints
        # obs_state = np.array([self.controller.obstacle_obs.x,
        #                       self.controller.obstacle_obs.y,
        #                       self.controller.obstacle_obs.theta,
        #                       self.controller.obstacle_obs.vx,
        #                       self.controller.obstacle_obs.delta])
        # trajectory_constraints = self.visualize_constraints(timestamps_cons_num, obs_state)
        # trajectories += [t for t in trajectory_constraints]
        #
        # visualize target region
        trajectories += [self.visualize_ref()]
        #
        # # trajectories += [t for t in self.visualize_all_constraints(timestamps_cons_num)]

        return trajectories

    def get_future_state(self, time_step: int) -> VehicleState:
        x_t = self.last_output["x"][:, time_step]
        future_state = VehicleState(x=x_t[0], y=x_t[1], theta=x_t[2], vx=x_t[3], delta=x_t[4])
        return future_state

    def visualize_MIP_plan(self):
        future_states = [self.my_state]
        timestamps_MIP = [0]
        for time_step in range(self.model.N):
            future_states += [self.get_future_state(time_step)]
            timestamps_MIP += [time_step + 1]
        trajectory_MIP = (DgSampledSequence[VehicleState](timestamps_MIP, values=future_states), 'gold')
        return trajectory_MIP

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
