from typing import Optional

from geometry import translation_from_SE2

from dg_commons.planning.trajectory import Trajectory
from sim.agents.lane_follower import LFAgent
from sim.models.vehicle import VehicleState
from sim.sim_vis_extra import DrawableTrajectoryType

__all__ = ["B1Agent"]


class B1Agent(LFAgent):
    """Baseline 1 agent"""

    def on_get_extra(self, ) -> Optional[DrawableTrajectoryType]:
        if not self._emergency:
            return None
        mypose = self.pure_pursuit.pose()
        p, theta = translation_from_SE2(mypose)
        traj = Trajectory(
            timestamps=[0, 1],
            values=[VehicleState(x=p[0], y=p[1], theta=theta, vx=0, delta=0),
                    VehicleState(x=p[0], y=p[1], theta=theta, vx=0, delta=0),
                    ])
        traj_s = [traj, ]
        colors = ["gold", ]
        return list(zip(traj_s, colors))
