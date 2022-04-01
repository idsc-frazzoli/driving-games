import random
from decimal import Decimal as D
from typing import Optional

from dg_commons import U, PlayerName, logger
from dg_commons.maps import DgLanelet
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations

__all__ = ["StopOrGoAgent"]


class StopOrGoAgent(LFAgent):
    """
    Stop or Go agent. Continues to follow reference with prob=prob_go, stops with prob = 1-prob_go.
    It uses the logic of a lane following agent.
    :param ref_lane: reference lane to follow
    :param stopping_time: simulation time at which vehicle should come to a complete halt
    :param behavior: behavior can be specified as 'stop' or 'go'. If anything else is given, random sampling.
    :param prob_go: probability than the agent should go. 1-prob_go is probability of stopping
    """

    def __init__(self,
                 ref_lane: DgLanelet,
                 stopping_time: D = D(0),
                 behavior: str = None,
                 prob_go: float = 0.5):

        super().__init__()
        self.ref_lane = ref_lane
        assert 0 <= prob_go <= 1, "Probability of going must be in range [0,1]"
        self.prob_go = prob_go
        if behavior is None:
            self.behavior: str = "undefined"
        else:
            assert behavior in ["stop", "go"], "Behavior can only be stop or go."
            self.behavior = behavior

        self.stopping_time = D(stopping_time)

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        super().on_episode_init(init_sim_obs=init_sim_obs)
        if self.behavior == "undefined":
            # sample if the player should go or should stop
            # random.seed(init_sim_obs.seed)
            unif = random.uniform(0, 1)
            if unif > self.prob_go:
                self.behavior = "stop"
            else:
                self.behavior = "go"

        logger.info("The behavior of the agent is: " + self.behavior)

    def get_commands(self, sim_obs: SimObservations) -> U:

        if self.behavior == "stop":
            if float(self.stopping_time) == 0:
                self.speed_behavior.params.nominal_speed = 0.0
            else:
                self.speed_behavior.params.nominal_speed \
                    = self.speed_behavior.params.nominal_speed \
                      * float((self.stopping_time - sim_obs.time)) / float(self.stopping_time)

        if self.speed_behavior.params.nominal_speed < 0.0:
            self.speed_behavior.params.nominal_speed = 0.0

        elif self.behavior == "go":
            pass

        commands = super().get_commands(sim_obs)

        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        pass
