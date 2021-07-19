from typing import Mapping

from dataclasses import dataclass

from games import PlayerName
from sim import logger
from sim.agent import Agent
from sim.simulator_structures import *


@dataclass
class SimContext:
    # todo need to add the map
    models: Mapping[PlayerName, SimModel]
    players: Mapping[PlayerName, Agent]
    time: SimTime
    log: SimulationLog
    param: SimParameters
    sim_terminated: bool
    seed: int

    def __post_init__(self):
        assert all([player in self.models for player in self.players])


class Simulator:
    last_observations: SimObservations

    def pre_update(self, sim_context: SimContext):
        self.last_observations.time = sim_context.time
        self.last_observations.players = {}
        for player_name, model in sim_context.models.items():
            self.last_observations.players.update({player_name: model.state})
        logger.debug_print(f"Pre update function, sim time {sim_context.time}")
        logger.debug_print(f"Last observations:\n{self.last_observations}")

    def update(self, sim_context: SimContext):
        sim_context.log[sim_context.time] = {}
        # fixme this can be parallelized later
        for player_name, model in sim_context.models.items():
            actions = sim_context.players[player_name].get_commands(self.last_observations)
            new_state = model.update(actions, dt=sim_context.param.dt)
            log_entry = LogEntry(state=new_state, actions=actions)
            logger.debug_print(f"Update function, sim time {sim_context.time}, player: {player_name}")
            logger.debug_print(f"From {model.state} to {new_state} applying {actions}")
            sim_context.log[sim_context.time].update({player_name: log_entry})
        # todo check if sim context gets updates properly or it needs to be returned

    def post_update(self, sim_context: SimContext):

        # todo here we check for collisions and termination (end of sim time and so on)

        sim_context.time += sim_context.param.dt
        if sim_context.time > 5:  # todo temporary we just simulate 5 seconds
            sim_context.sim_terminated = True

        ...
