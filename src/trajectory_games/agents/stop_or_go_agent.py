import random
from decimal import Decimal as D
from typing import Optional

from dg_commons import U, PlayerName, logger
from dg_commons.maps import DgLanelet
from dg_commons.sim import DrawableTrajectoryType
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.simulator_structures import SimObservations

__all__ = ["StopOrGoAgent"]


class StopOrGoAgent(LFAgent):
    """
    Stop or Go agent. Goes straight across intersection with prob=prob_go, stops with prob = 1-prob_go.
    :param ref_lane: refernce lanenelet to follow
    :param stopping_time: simulation time at which vehicle should come to a complete halt
    :param behavior: behavior can be specified as 'stop' or 'go'. If anything else is given, random sampling.
    :param seed: seed for random sampler.
    :param prob_go: probability than the agent should go. 1-prob_go is probability of stopping
    """

    def __init__(self,
                 ref_lane: DgLanelet,
                 stopping_time: D = 0.0,
                 behavior: str = None,
                 seed: int = 0,
                 prob_go: float = 0.5):

        super().__init__()
        self.ref_lane = ref_lane
        self.seed = seed
        assert 0 <= prob_go <= 1, "Probability of going must be in range [0,1]"
        self.prob_go = prob_go
        if behavior is None:
            self.behavior: str = "undefined"
        else:
            assert behavior in ["stop", "go"], "Behavior can only be stop or go."
            self.behavior = behavior

        self.stopping_time = stopping_time

    def on_episode_init(self, my_name: PlayerName):
        super().on_episode_init(my_name=my_name)
        if self.behavior not in ['go', 'stop']:
            # sample if the player should go or should stop
            random.seed(a=self.seed)
            unif = random.uniform(0, 1)
            if unif > self.prob_go:
                self.behavior = "stop"
            else:
                self.behavior = "go"

        logger.info("The behavior of the agent is: " + self.behavior)

    def get_commands(self, sim_obs: SimObservations) -> U:

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

        return commands

    def on_get_extra(
            self,
    ) -> Optional[DrawableTrajectoryType]:
        pass
