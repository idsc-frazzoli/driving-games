from typing import Optional

from geometry import SE2_from_xytheta, SE2value

from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim import SimObservations
from sim.agents.agent import Agent
from sim.models.vehicle import VehicleCommands


class LFAgent(Agent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a pure pursuit controller. The reference in speed is determined by the speed behavior.
    """

    def __init__(self, lane: DgLanelet,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 pure_pursuit: Optional[PurePursuit] = None):
        self.ref_lane = lane
        self.speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        self.speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.pure_pursuit: PurePursuit = PurePursuit() if pure_pursuit is None else pure_pursuit
        self.my_name: Optional[PlayerName] = None

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        self.pure_pursuit.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        my_obs = sim_obs.players[self.my_name]
        my_pose: SE2value = SE2_from_xytheta([my_obs.x, my_obs.y, my_obs.theta])

        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_observations(current_speed=my_obs.vx)
        lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
        self.pure_pursuit.update_pose(pose=my_pose, along_path=lanepose.along_lane)

        # compute commands
        t = float(sim_obs.time)
        speed_ref = self.speed_behavior.get_speed_ref(t)
        self.pure_pursuit.update_speed(speed=speed_ref)
        self.speed_controller.update_reference(desired_speed=speed_ref)
        acc = self.speed_controller.get_control(t)
        # pure proportional with respect to delta error
        kp = 0.5
        ddelta = kp * (self.pure_pursuit.get_desired_steering() - my_obs.delta)
        return VehicleCommands(
            acc=acc,
            ddelta=ddelta
        )
