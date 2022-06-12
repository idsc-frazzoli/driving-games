import random
import sys
from datetime import datetime
from functools import partial
from itertools import product
from typing import Optional, Mapping, FrozenSet, Any

from dg_commons import U, PlayerName, logger, DgSampledSequence, iterate_dict_combinations
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations
from dg_commons.time import time_function
from possibilities import ProbDist
from preferences import Preference
from trajectory_games import MetricEvaluation, \
    TrajectoryWorld
from trajectory_games.stochastic_decision_making.uncertain_preferences import UncertainPreferenceOutcomes
from trajectory_games.structures import TrajectoryGamePosetsParam

__all__ = ["UncertainOutcomeAgent"]


def save_action_outcome_mapping(obj, filename):
    """
    For result inspection. Generates .txt file with joint actions and joint outcomes
    """
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


class UncertainOutcomeAgent(Agent):
    """Agent selecting optimal action with stochastic outcomes. Implemented for a setting with 2 players, where
    the name of the other agent is "OTHER" """

    def __init__(
            self,
            my_name: PlayerName,
            pref_distr: Mapping[PlayerName, ProbDist[Preference]],
            ego_pref: Preference,
            game_params: TrajectoryGamePosetsParam,
            world: TrajectoryWorld,
            action_selection_method: str,
            trajectories_and_commands: Mapping[PlayerName, Mapping[Trajectory, Trajectory]],
            **kwargs
    ):

        self.game_params = game_params
        self.my_name: PlayerName = my_name
        assert self.my_name in trajectories_and_commands.keys()
        for player_name in trajectories_and_commands.keys():
            if player_name == self.my_name:
                continue
            self.other_name = player_name

        self.pref_distr = pref_distr
        self.ego_pref = ego_pref
        self.world = world
        self.trajectories_and_commands = trajectories_and_commands
        self.action_selection_method = action_selection_method

        self.joint_actions_outcomes_mapping = {}

        # for control
        self.commands: Optional[DgSampledSequence[VehicleCommands]] = None
        self.trajectory: Optional[Trajectory] = None

        # for plotting
        self.all_selected_trajectories = None
        self.all_trajectories: Mapping[PlayerName, FrozenSet[Trajectory]] = {}

    def generate_trajs_and_compute_outcomes(self):
        # generate trajectories for Ego with trajectory generator
        #     ego_traj_gen = TrajectoryGenerator(params=self.game_params.traj_gen_params[self.my_name],
        #                                        ref_lane_goals=self.game_params.ref_lanes[self.my_name])
        #
        #     graph = ego_traj_gen.get_actions(state=self.game_params.initial_states[self.my_name], return_graphs=True)
        #     self.ego_traj_graph, = graph
        #     actions_ego_all = self.ego_traj_graph.get_all_transitions()

        actions_ego_all = self.trajectories_and_commands[self.my_name]
        other_agent_actions = set(self.trajectories_and_commands[self.other_name].keys())

        self.all_trajectories[self.my_name] = set(actions_ego_all.keys())
        self.all_trajectories[self.other_name] = other_agent_actions

        get_outcomes = partial(MetricEvaluation.evaluate, world=self.world)

        for joint_traj in set(iterate_dict_combinations(self.all_trajectories)):
            self.joint_actions_outcomes_mapping[joint_traj] = get_outcomes(joint_traj)

        return

    def select_action_uncertain_outcomes_prefs(self, store_action_outcome_mapping=False):
        """
        compute optimal action with uncertain outcomes
        :return:
        """
        # write to .txt file for inspection later
        if store_action_outcome_mapping:
            now_str = datetime.now().strftime("%y-%m-%d-%H%M%S") + str(".txt")
            output_dir = "/home/leon/Documents/repos/driving-games/src/trajectory_games_tests/experiments/" + now_str
            save_action_outcome_mapping(obj=self.joint_actions_outcomes_mapping, filename=output_dir)

        uncertain_preference_outcomes = UncertainPreferenceOutcomes(
            my_name=self.my_name,
            pref_distr=self.pref_distr,
            joint_actions_outcomes_mapping=self.joint_actions_outcomes_mapping,
            ego_pref=self.ego_pref)

        trajectories = uncertain_preference_outcomes.action_selector(
            method=self.action_selection_method)
        if len(trajectories) > 1:
            logger.info(
                "Sampled 1 optimal action out of " + str(len(trajectories)) + " available optimal trajectories.")
        self.all_selected_trajectories = trajectories
        self.trajectory = random.sample(trajectories, 1)[0]

        self.commands = self.trajectories_and_commands[self.my_name][self.trajectory]

    @time_function
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.generate_trajs_and_compute_outcomes()
        self.select_action_uncertain_outcomes_prefs()

    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        commands = self.commands.at_interp(current_time)
        # hard coded to avoid driving backwards
        if sim_obs.players[self.my_name].state.vx < 0:
            commands.acc = 0.5
        return commands

    def on_get_extra(self) -> Optional[Any]:
        trajectories = self.all_trajectories[self.my_name]

        candidates = tuple(
            product(
                trajectories,
                [
                    "indianred",
                ],
            )
        )
        selected_trajectories = tuple(
            product(
                self.all_selected_trajectories,
                [
                    "darkred",
                ],
            )
        )

        candidates += selected_trajectories

        return candidates
