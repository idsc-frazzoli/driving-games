from typing import Optional

from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons import DgSampledSequence, U, PlayerName, X
from dg_commons.sim.simulator_structures import SimObservations
from geometry import translation_from_SE2

__all__ = ["GamePlayingAgent"]


class GamePlayingAgent(Agent):
    """Baseline 1 agent"""

    def __init__(self):

        self.trajectory = None

    def on_episode_init(self, my_name: PlayerName):
        pass

    def get_commands(self, sim_obs: SimObservations) -> U:
        """This method gets called for each player inside the update loop of the simulator"""
        pass

    def on_get_extra(
        self,
    ) -> Optional[DrawableTrajectoryType]:
        if not self._emergency:
            return None
        mypose = self.pure_pursuit.pose
        p = translation_from_SE2(mypose)
        traj = Trajectory(
            timestamps=[0, 1],
            values=[
                VehicleState(x=p[0], y=p[1], theta=0, vx=0, delta=0),
                VehicleState(x=p[0], y=p[1], theta=0, vx=0, delta=0),
            ],
        )
        traj_s = [
            traj,
        ]
        colors = [
            "gold",
        ]
        return list(zip(traj_s, colors))
