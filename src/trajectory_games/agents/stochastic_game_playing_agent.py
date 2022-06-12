import copy
import random
import sys
from datetime import datetime
from functools import partial
from itertools import product
from typing import Optional, Mapping, FrozenSet, Any

import numpy as np

from dg_commons import U, PlayerName, DgSampledSequence, iterate_dict_combinations, logger
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations
from dg_commons.time import time_function
from possibilities import ProbDist
from preferences import Preference
from trajectory_games import SolvedGameNode, TrajectoryGenerator, MetricEvaluation, \
    TrajectoryWorld
from trajectory_games.bicycle_dynamics import BicycleDynamics
from trajectory_games.stochastic_decision_making.posetal_game_with_uncertainty import StochasticNEComputation
from trajectory_games.structures import TrajectoryGamePosetsParam, TrajectoryGenParams

__all__ = ["StochasticGamePlayingAgent"]
P1 = PlayerName('P1')


def save_action_outcome_mapping(obj, filename):
    file_path = filename
    sys.stdout = open(file_path, "w")
    # print keys (joint actions)
    print("ACTIONS" + "\n")
    for idx, key in enumerate(obj.keys()):
        print("Joint action # " + str(idx) + " :")
        for pname, traj in key.items():
            print(pname + ": " + str(traj) + "\n")
            # print values (joint outcomes)
        print("\n")

    print("\n")
    print("OUTCOMES" + "\n")
    for idx, val in enumerate(obj.values()):
        print("Joint outcome # " + str(idx) + " :")
        for pname, metric_dict in val.items():
            print(pname + ": ")
            for metric, eval_metric in metric_dict.items():
                print(str(metric.get_name()) + ": " + str(eval_metric.value))
            print("\n")
        print("\n")


def get_stop_or_go_trajectories(initial_state: VehicleState, stopping_time: float, params: TrajectoryGenParams):
    bicycle_dyn = BicycleDynamics(params=params)

    assert stopping_time != 0.0
    acc_stop = -initial_state.vx / float(stopping_time)
    acc_go = 0.0

    dst = 0.0
    constant_commands = {"go": VehicleCommands(acc=acc_go, ddelta=dst),
                         "stop": VehicleCommands(acc=acc_stop, ddelta=dst)}

    dt = float(params.dt)
    max_time = dt * (params.max_gen - 1)
    trajs_and_commands = {}
    for u in constant_commands.values():
        samp_traj = []
        current_state = initial_state
        values_traj = [current_state]
        timesteps_traj = [0.0]
        commands_traj = [u]
        for time in np.arange(0, max_time, dt):
            # don't allow rear driving
            if current_state.vx < 0.0:
                u.acc = 0.0

            # feasible_acc = bicycle_dyn.get_feasible_acc(x=current_state, dt=params.dt, u_acc=[u.acc])
            # u.acc = feasible_acc.pop()

            next_state, samp_states = bicycle_dyn.successor_ivp(
                x0=(time, current_state), u=u, dt=params.dt, dt_samp=params.dt_samp
            )
            current_state = next_state[1]
            samp_traj = samp_traj + samp_states[1:]

            for _ in range(len(samp_states[1:])):
                commands_traj.append(copy.deepcopy(u))

        for tup in samp_traj:
            values_traj.append(tup[1])
            timesteps_traj.append(tup[0])

        traj = Trajectory(values=values_traj, timestamps=timesteps_traj)
        commands = Trajectory(values=commands_traj, timestamps=timesteps_traj)
        trajs_and_commands[traj] = commands

    return trajs_and_commands


class StochasticGamePlayingAgent(Agent):
    """Agent solving a trajectory game while holding a belief over the type of other agents"""

    def __init__(
            self,
            my_name: PlayerName,
            pref_distr: Mapping[PlayerName, ProbDist[Preference]],
            game_params: TrajectoryGamePosetsParam,
            world: TrajectoryWorld,
            **kwargs
    ):

        self.actions_commands_ego_all = None
        self.game_params = game_params
        self.my_name: PlayerName = my_name

        self.joint_actions_outcomes_mapping = {}

        self.pref_distr = pref_distr
        self.world = world
        self.other_agent = "stop_or_go"

        self.uncertain_preference_outcomes = None

        # for control
        self.commands: Optional[DgSampledSequence[VehicleCommands]] = None
        self.trajectory: Optional[Trajectory] = None

        # for plotting
        self.all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}
        self.selected_eq: Optional[SolvedGameNode] = None
        if kwargs["other_stopping_time"]:
            self.other_stopping_time = kwargs["other_stopping_time"]
        else:
            self.other_stopping_time = None

    def generate_trajs_and_compute_outcomes(self):
        # using trajectory generator
        # ego_traj_gen = TrajectoryGenerator(params=self.game_params.traj_gen_params[self.my_name],
        #                                    ref_lane_goals=self.game_params.ref_lanes[self.my_name])

        # here maybe use state from SimContext
        # graph = ego_traj_gen.get_actions(state=self.game_params.initial_states[self.my_name], return_graphs=True)
        # self.ego_traj_graph, = graph

        # actions_ego_all = self.ego_traj_graph.get_all_transitions()

        # generating custom trajectories for experiment
        self.actions_commands_ego_all = get_stop_or_go_trajectories(
            initial_state=self.game_params.initial_states[self.my_name],
            params=self.game_params.traj_gen_params[self.my_name],
            stopping_time=2.0)

        # discard trajectories that did not reach the end
        final_time = (self.game_params.traj_gen_params[self.my_name].max_gen - 1) * \
                     float(self.game_params.traj_gen_params[self.my_name].dt)
        actions_to_remove = set()
        for action in self.actions_commands_ego_all.keys():
            if action.timestamps[-1] != final_time:
                actions_to_remove.add(action)
        actions_ego_all = set(self.actions_commands_ego_all.keys()).difference(actions_to_remove)

        random.seed(0)
        actions_ego = random.sample(actions_ego_all, 2)  # sample some actions
        # todo: here get actions of stop or go agent
        if self.other_agent == "stop_or_go" and self.other_stopping_time is not None:
            other_agent_actions = get_stop_or_go_trajectories(initial_state=self.game_params.initial_states[P1],
                                                              stopping_time=self.other_stopping_time,
                                                              params=self.game_params.traj_gen_params[P1])
            # other_agent_actions = random.sample(actions_ego_all, 1)
        else:
            raise NotImplementedError

        self.all_trajectories[self.my_name] = actions_ego
        self.all_trajectories[P1] = other_agent_actions

        get_outcomes = partial(MetricEvaluation.evaluate, world=self.world)

        for joint_traj in set(iterate_dict_combinations(self.all_trajectories)):
            self.joint_actions_outcomes_mapping[joint_traj] = get_outcomes(joint_traj)

        return

    def compute_uncertain_NE(self):
        """
        compute NE from distribution over preferences
        :return:
        """
        # write to .txt file for inspection later
        now_str = datetime.now().strftime("%y-%m-%d-%H%M%S") + str(".txt")
        output_dir = "/home/leon/Documents/repos/driving-games/src/trajectory_games_tests/experiments/" + now_str
        save_action_outcome_mapping(obj=self.joint_actions_outcomes_mapping, filename=output_dir)

        stoch_NE_comp = StochasticNEComputation(pref_distr=self.pref_distr,
                                                joint_actions_outcomes_mapping=self.joint_actions_outcomes_mapping)

        uncertain_NE = stoch_NE_comp.NE_distr()
        return uncertain_NE

    @time_function
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # random.seed(init_sim_obs.seed)
        self.generate_trajs_and_compute_outcomes()

        uncertain_NE = self.compute_uncertain_NE()

        most_likely_eqs = max(uncertain_NE.p, key=uncertain_NE.p.get)
        strong_NE = most_likely_eqs["strong_eqs"]

        if len(strong_NE) > 1:
            logger.info("More than one strong equilibrium found: ")
            logger.info(strong_NE)
            logger.info("Sampling one at random.")

        sampled_strong_NE = random.sample(strong_NE, 1)[0]

        self.trajectory = sampled_strong_NE[self.my_name]
        self.commands = self.actions_commands_ego_all[self.trajectory]

    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        return self.commands.at_interp(current_time)

    def on_get_extra(self) -> Optional[Any]:
        # store metrics in extra of player logger
        # if self.game_params.store_metrics:
        #     return self.metric_violation

        # store trajectories in extra of player logger (or plotting)

        trajectories = self.all_trajectories[self.my_name]
        trajectories_blue = self.all_trajectories[P1]
        # selected_trajectory_blue = self.selected_eq.actions[P1]
        # todo: here select TRUE trajectory BLUE
        selected_traj = self.trajectory
        candidates = tuple(
            product(
                trajectories,
                [
                    "indianred",
                ],
            )
        )
        new_tuple_red = (selected_traj, 'darkred')
        candidates += (new_tuple_red,)

        candidates_blue = tuple(
            product(
                trajectories_blue,
                [
                    "cornflowerblue",
                ],
            )
        )
        candidates += candidates_blue

        # new_tuple_blue = (selected_trajectory_blue, 'mediumblue')
        # candidates += (new_tuple_blue,)

        return candidates
