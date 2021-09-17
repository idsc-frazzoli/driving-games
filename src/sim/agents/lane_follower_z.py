from typing import Tuple, get_args
from dg_commons.controllers.controller_types import *
from typing import Optional
from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.planning.lanes import DgLanelet
from games import PlayerName, X
from sim import SimObservations
from sim.agents.agent import Agent
from sim.models.vehicle import VehicleCommands


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
                 steering_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):

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

        self.my_name: Optional[PlayerName] = None
        self.return_extra: bool = return_extra
        self._emergency: bool = False
        self._my_obs: Optional[X] = None

        self.state_estimator = None
        self.commands = None

    def set_state_estimator(self, state_estimator):
        self.state_estimator = state_estimator

    def measurement_update(self, measurement):
        if self.state_estimator is not None:
            self.state_estimator.update_prediction(self.commands)
            self.state_estimator.update_measurement(measurement)
            self.state = self.state_estimator.state
        else:
            self.state = measurement

    @staticmethod
    def get_default_la(lane: DgLanelet):
        return LFAgent(lane, PurePursuit(), SpeedBehavior(), SpeedController(), SteeringController())

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        self.controller.update_path(self.ref_lane)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        my_obs = sim_obs.players[self.my_name]

        t = float(sim_obs.time)
        self.speed_behavior.update_observations(sim_obs.players)
        speed_ref, emergency = self.speed_behavior.get_speed_ref(t)

        if emergency or self._emergency:
            # Once the emergency kicks in the speed ref will always be 0
            self._emergency = True
            speed_ref = 0
            self.emergency_subroutine()

        if self.decoupled:
            acc, ddelta = self._get_decoupled_commands(my_obs, speed_ref, t)
        else:
            acc, ddelta = self._get_coupled_commands(my_obs, speed_ref, t)

        self.commands = VehicleCommands(acc=acc, ddelta=ddelta)
        return self.commands

    def _get_decoupled_commands(self, my_obs: X, speed_ref: float, t: float) -> Tuple[float, float]:

        # update observations
        self.speed_controller.update_measurement(measurement=my_obs.vx)
        self.controller.update_state(my_obs)
        # compute commands
        self.speed_controller.update_reference(reference=speed_ref)
        acc = self.speed_controller.get_control(t)

        ddelta = self.steering_controller.get_steering_velocity(self.controller.get_desired_steering(), my_obs.delta)

        return acc, ddelta

    def _get_coupled_commands(self, my_obs: X, speed_ref: float, t: float) -> Tuple[float, float]:
        # compute commands
        self.controller.update_state(my_obs, speed_ref)

        steering, acc = self.controller.get_targets()
        ddelta = self.steering_controller.get_steering_velocity(steering, my_obs.delta)
        return acc, ddelta

    def emergency_subroutine(self) -> VehicleCommands:
        pass
