from typing import Optional, Tuple, get_args

import numpy as np
from geometry import SE2_from_xytheta, SE2value
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName
from sim import SimObservations, logger
from sim.agents.agent import Agent
from sim.models.vehicle import VehicleCommands
from dg_commons.controllers.controller_types import *
from games import X


class LFAgent(Agent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a lateral controller. The reference speed is determined by the speed behavior and it
    is tracked by a speed controller.
    """

    def __init__(self,
                 lane: DgLanelet,
                 controller: Union[LateralController, LatAndLonController],
                 speed_behavior: Optional[LongitudinalBehavior] = None,
                 speed_controller: Optional[LongitudinalController] = None,
                 steering_controller: Optional[SteeringController] = None):

        decoupled: bool = type(controller) in get_args(LateralController) and type(speed_controller) in get_args(LongitudinalController)
        single: bool = type(controller) in get_args(LatAndLonController) and speed_controller is None
        assert decoupled or single

        self.ref_lane = lane
        self.my_name: Optional[PlayerName] = None
        self.decoupled = decoupled

        self.controller: Union[LateralController, LatAndLonController] = controller
        self.speed_controller: Optional[LongitudinalController] = speed_controller
        self.speed_behavior: LongitudinalBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.steering_controller: SteeringController = SCP() if steering_controller is None else steering_controller

    @classmethod
    def get_default_la(cls, lane: DgLanelet):
        pass

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        self.controller.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        my_obs = sim_obs.players[self.my_name]
        t = float(sim_obs.time)
        self.speed_behavior.update_observations(sim_obs.players)

        if self.decoupled:
            acc, ddelta = self._get_decoupled_commands(my_obs, t)
        else:
            acc, ddelta = self._get_coupled_commands(my_obs, t)

        if not -1 <= ddelta <= 1:
            logger.info(f"Agent {self.my_name}: clipping ddelta: {ddelta} within [-1,1]")
            ddelta = np.clip(ddelta, -1, 1)
        return VehicleCommands(
            acc=acc,
            ddelta=ddelta
        )

    def _get_decoupled_commands(self, my_obs: X, t: float) -> Tuple[float, float]:

        # update observations
        self.speed_controller.update_measurement(measurement=my_obs.vx)
        self.controller.update_state(my_obs)
        # compute commands
        speed_ref = self.speed_behavior.get_speed_ref(t)
        self.speed_controller.update_reference(reference=speed_ref)
        acc = self.speed_controller.get_control(t)

        ddelta = self.steering_controller.get_steering_velocity(self.controller.get_desired_steering(), my_obs.delta)

        return acc, ddelta

    def _get_coupled_commands(self, my_obs: X, t: float) -> Tuple[float, float]:
        # update observations
        self.controller.update_state(my_obs)
        # compute commands
        speed_ref = self.speed_behavior.get_speed_ref(t)
        self.controller.update_ref_speed(speed_ref)

        acc, steering = self.controller.get_targets()
        ddelta = self.steering_controller.get_steering_velocity(steering, my_obs.delta)
        return acc, ddelta
