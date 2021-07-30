from decimal import Decimal
from itertools import combinations
from typing import Mapping, Optional

from dataclasses import dataclass, field

from duckietown_world import DuckietownMap

from games import PlayerName
from sim import logger
from sim.agent import Agent
from sim.simulator_structures import *
from world import load_driving_game_map


@dataclass
class SimContext:
    map_name: str
    map: DuckietownMap = field(init=False)
    models: Mapping[PlayerName, SimModel]
    players: Mapping[PlayerName, Agent]
    log: SimulationLog
    param: SimParameters
    time: SimTime = Decimal(0)
    seed: int = 0
    sim_terminated: bool = False

    def __post_init__(self):
        assert all([player in self.models for player in self.players])
        self.map = load_driving_game_map(self.map_name)


# todo for now just a bool in the future we want more detailed info
CollisionReport = bool


class Simulator:
    last_observations: Optional[SimObservations] = SimObservations(players={}, time=Decimal(0))

    def run(self, sim_context: SimContext):
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)

    def pre_update(self, sim_context: SimContext):
        self.last_observations.time = sim_context.time
        self.last_observations.players = {}
        for player_name, model in sim_context.models.items():
            self.last_observations.players.update({player_name: model.get_state()})
        logger.debug(f"Pre update function, sim time {sim_context.time:.2f}")
        logger.debug(f"Last observations:\n{self.last_observations}")
        return

    def update(self, sim_context: SimContext):
        sim_context.log[sim_context.time] = {}
        # fixme this can be parallelized later
        for player_name, model in sim_context.models.items():
            actions = sim_context.players[player_name].get_commands(self.last_observations)
            model.update(actions, dt=sim_context.param.dt)
            log_entry = LogEntry(state=model.get_state(), actions=actions)
            logger.debug(f"Update function, sim time {sim_context.time:.2f}, player: {player_name}")
            logger.debug(f"New state {model.get_state()} reached applying {actions}")
            sim_context.log[sim_context.time].update({player_name: log_entry})
        return

    def post_update(self, sim_context: SimContext):
        collision_report = self._check_collisions(sim_context)
        sim_context.time += sim_context.param.dt
        if sim_context.time > sim_context.param.max_sim_time or collision_report:
            sim_context.sim_terminated = True
        return

    @staticmethod
    def _check_collisions(sim_context: SimContext) -> CollisionReport:
        """
        This checks only collision at the current step, tunneling effects and similar are ignored
        :param sim_context:
        :return:
        """
        collision = False
        for a, b in combinations(sim_context.models, 2):
            a_shape = sim_context.models[a].get_footprint()
            b_shape = sim_context.models[b].get_footprint()
            collision = a_shape.collide(b_shape) or collision
            if collision:
                logger.info(f"Detected a collision between {a} and {b}, Terminating simulation")
        return collision
