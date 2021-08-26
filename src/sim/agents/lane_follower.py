from typing import Optional

import numpy as np
from geometry import SE2_from_xytheta, SE2value
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim import SimObservations, logger
from sim.agents.agent import Agent
from sim.models.vehicle import VehicleCommands
from dg_commons.controllers.controller_types import *


class LFAgent(Agent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a lateral controller. The reference speed is determined by the speed behavior and it
    is tracked by a speed controller.
    """

    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[LongitudinalController] = None,
                 speed_behavior: Optional[LongitudinalBehavior] = None,
                 lateral_controller: Optional[LateralController] = None,
                 steering_controller: Optional[SteeringController] = None):
        self.ref_lane = lane
        self.speed_controller: LongitudinalController = SpeedController() if speed_controller is None else speed_controller
        self.speed_behavior: LongitudinalBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.lateral_controller: LateralController = PurePursuit() if lateral_controller is None else lateral_controller
        self.steering_controller: SteeringController = SCP() if steering_controller is None else steering_controller
        self.my_name: Optional[PlayerName] = None

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        self.lateral_controller.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        my_obs = sim_obs.players[self.my_name]
        my_pose: SE2value = SE2_from_xytheta([my_obs.x, my_obs.y, my_obs.theta])

        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_measurement(measurement=my_obs.vx)
        lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
        self.lateral_controller.update_pose(pose=my_pose, along_path=lanepose.along_lane)

        # compute commands
        t = float(sim_obs.time)
        speed_ref = self.speed_behavior.get_speed_ref(t)
        self.lateral_controller.update_speed(speed=speed_ref)
        self.speed_controller.update_reference(reference=speed_ref)
        acc = self.speed_controller.get_control(t)
        # pure proportional with respect to delta error

        ddelta = self.steering_controller.get_steering_velocity(self.lateral_controller.get_desired_steering(), my_obs.delta)

        if not -1 <= ddelta <= 1:
            logger.info(f"Agent {self.my_name}: clipping ddelta: {ddelta} within [-1,1]")
            ddelta = np.clip(ddelta, -1, 1)
        return VehicleCommands(
            acc=acc,
            ddelta=ddelta
        )
