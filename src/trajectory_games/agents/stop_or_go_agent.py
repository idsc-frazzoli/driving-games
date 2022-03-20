import random
from decimal import Decimal as D
from os.path import exists
from typing import Optional, List

from dg_commons import U, PlayerName, logger
from dg_commons.maps import DgLanelet
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import DrawableTrajectoryType, SimTime
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.simulator_structures import SimObservations

__all__ = ["StopOrGoAgent", "read_traj", "write_traj"]

import pickle


def write_traj(file_path: str, behavior: str, states: Trajectory, commands: Trajectory):
    logger.info("Writing trajectories to Pickle file.")
    file_exists = exists(file_path)
    if not file_exists:
        mode = 'wb'
    else:
        mode = 'ab'

    to_dump = {behavior: {"trajectory": states, "commands": commands}}
    with open(file_path, mode) as f:
        pickle.dump(to_dump, f)


def read_traj(file_path: str, behavior: str):
    assert behavior in ["stop", "go"], "Behavior can only be stop or go."
    logger.info("Reading trajectories from Pickle file.")
    with open(file_path, 'rb') as f:
        traj = pickle.load(f)
        # if the first trajectory is the required one, return it
        if list(traj.keys())[0] == behavior:
            return traj[list(traj.keys())[0]]
        # else, return the other trajectory
        else:
            traj = pickle.load(f)  # load next item of dictionary
            return traj[list(traj.keys())[0]]


class StopOrGoAgent(LFAgent):
    """Stop or Go agent. Goes straight across intersection with prob=prob_go, stops with prob = 1-prob_go.
    When agent used in generative mode, it creates and stores the trajectories for starting or stopping.
    When generative is False, the trajectories are loaded from a stored file."""

    #todo: when doing N experiments, look into the seed issue a bit more.
    def __init__(self,
                 ref_lane: DgLanelet,
                 max_sim_time: SimTime = None,
                 stopping_time: D = None,
                 file_path: str = None,
                 behavior: str = None,
                 generative: bool = False,
                 seed: int = 0,
                 prob_go: float = 0.5):

        super().__init__()
        self.ref_lane = ref_lane
        self.seed = seed
        assert 0 <= prob_go <= 1, "Probability of going must be in range [0,1]"
        self.p = prob_go
        if behavior is None:
            self.behavior: str = "undefined"
        else:
            assert behavior in ["stop", "go"], "Behavior can only be stop or go."
            self.behavior = behavior
        self.generative = generative
        self.max_sim_time = max_sim_time
        if generative:
            assert stopping_time is not None and max_sim_time is not None, \
                "You need to provide a stopping time and a max_sim_time for the generative model."
        self.stopping_time = stopping_time
        self._states: List[VehicleState] = []
        self._commands: List[VehicleCommands] = []
        self._timestamps: List[SimTime] = []
        if file_path is None:
            self.file_path: str = "trajectory_stop_go.pickle"
        else:
            self.file_path = file_path
        self.trajectory: Trajectory = Trajectory([], [])
        self.commands: Trajectory = Trajectory([], [])

    def on_episode_init(self, my_name: PlayerName):
        super().on_episode_init(my_name=my_name)
        if self.behavior == "undefined":
            # sample if the player should go or should stop
            random.seed(a=self.seed)
            unif = random.uniform(0, 1)
            if unif > self.p:
                self.behavior = "stop"
            else:
                self.behavior = "go"

        logger.info("The behavior of the agent is: " + self.behavior)

        if not self.generative:
            logger.info("Not using generative model, will load data from pickle file.")
            # load trajectory from pre-computed trajectory
            trajectory_and_commands = read_traj(self.file_path, self.behavior)
            self.commands = trajectory_and_commands["commands"]
            self.trajectory = trajectory_and_commands["trajectory"]

    def get_commands(self, sim_obs: SimObservations) -> U:
        # Running in generative mode, a trajectory gets generated and stored
        # Use this to generate a trajectory that is realistic and feasible and can be used in the trajectory game

        if self.generative:
            logger.info("Generating trajectories.")
            if self.behavior == "stop":
                self.speed_behavior.params.nominal_speed \
                    = self.speed_behavior.params.nominal_speed \
                      * float((self.stopping_time - D(sim_obs.time)) / self.stopping_time)
                if self.speed_behavior.params.nominal_speed < 0.0:
                    self.speed_behavior.params.nominal_speed = 0.0

            elif self.behavior == "go":
                pass

            else:
                raise ValueError('Behavior can only be stop or go.')

            commands = super().get_commands(sim_obs)
            self._timestamps.append(sim_obs.time)
            self._states.append(sim_obs.players[self.my_name].state)
            self._commands.append(commands)

            # save trajectory and commands
            if sim_obs.time == self.max_sim_time:
                traj_states = Trajectory(values=self._states, timestamps=self._timestamps)
                traj_commands = Trajectory(values=self._commands, timestamps=self._timestamps)
                write_traj(file_path=self.file_path, behavior=self.behavior, states=traj_states, commands=traj_commands)

            return commands

        # if reading values from pre-computed trajectories
        else:
            return self.commands.at(sim_obs.time)

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        pass
