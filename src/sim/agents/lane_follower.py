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

    def __init__(self, lane: DgLanelet,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 pure_pursuit: Optional[PurePursuit] = None):
        self.ref_lane = lane
        self.speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        self.speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.pure_pursuit: PurePursuit = PurePursuit() if pure_pursuit is None else pure_pursuit
        self.my_name = None

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        my_obs = sim_obs.players[self.my_name]
        my_pose: SE2value = SE2_from_xytheta([my_obs.x, my_obs.y, my_obs.theta])
        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_observations(current_velocity=my_obs.vx)
        self.ref_lane.lane_pose_from_SE2_generic()
