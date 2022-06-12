from itertools import product
from typing import Optional, Set

from dg_commons import U
from dg_commons.planning import Trajectory
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations

__all__ = ["TrajectoryFollowingAgent"]


class TrajectoryFollowingAgent(Agent):
    """
    Agent that follows a prescribed trajectory with given commands.
    Alternative trajectories are optional alternatives passed for plotting.
    """

    def __init__(self,
                 trajectory: Trajectory,
                 commands: Trajectory,
                 alternative_trajectories: Optional[Set[Trajectory]] = None):

        self.trajectory = trajectory
        self.commands = commands
        self.alternative_trajectories = alternative_trajectories

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        pass

    def get_commands(self, sim_obs: SimObservations) -> U:
        current_time = sim_obs.time
        commands = self.commands.at_interp(current_time)

        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        trajectories_blue = self.alternative_trajectories
        selected_trajectory_blue = self.trajectory

        candidates_blue = tuple(
            product(
                trajectories_blue,
                [
                    "cornflowerblue",
                ],
            )
        )

        new_tuple_blue = (selected_trajectory_blue, 'mediumblue')
        candidates_blue += (new_tuple_blue,)

        return candidates_blue
