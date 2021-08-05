from dataclasses import dataclass, field
from decimal import Decimal
from itertools import permutations
from typing import Mapping, Optional, Dict

from duckietown_world import DuckietownMap

from games import PlayerName
from sim import logger, CollisionReport
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
    collision_reports: Dict[PlayerName, CollisionReport] = field(default_factory=dict)

    def __post_init__(self):
        assert all([player in self.models for player in self.players])
        self.map = load_driving_game_map(self.map_name)


class Simulator:
    last_observations: Optional[SimObservations] = SimObservations(players={}, time=Decimal(0))

    def run(self, sim_context: SimContext):
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)

    def pre_update(self, sim_context: SimContext):
        """Prior to stepping the simulation we compute the observations for each agent"""
        self.last_observations.time = sim_context.time
        self.last_observations.players = {}
        for player_name, model in sim_context.models.items():
            self.last_observations.players.update({player_name: model.get_state()})
        logger.debug(f"Pre update function, sim time {sim_context.time:.2f}")
        logger.debug(f"Last observations:\n{self.last_observations}")
        return

    def update(self, sim_context: SimContext):
        """ The real step of the simulation """
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
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        :param sim_context:
        :return:
        """
        collison_detected = self._check_collisions(sim_context)
        # after all the computations advance simulation time
        sim_context.time += sim_context.param.dt
        if sim_context.time > sim_context.param.max_sim_time or collison_detected:
            sim_context.sim_terminated = True
            # fixme simulation not stopping even when colliding
        return

    @staticmethod
    def _check_collisions(sim_context: SimContext) -> bool:
        """
        This checks only collision location at the current step, tunneling effects and similar are ignored
        :param sim_context:
        :return: True if at least one collision happened, False otherwise
        """
        collision = False
        # this way solves the permutations asymetrically
        for p1, p2 in permutations(sim_context.models, 2):
            a_shape = sim_context.models[p1].get_footprint()
            b_shape = sim_context.models[p2].get_footprint()
            if a_shape.intersects(b_shape):
                logger.info(f"Detected a collision between {p1} and {p2}")
                from sim.collision import compute_collision_report  # import here to avoid circular imports
                collision = True
                report: CollisionReport = compute_collision_report(p1, p2, sim_context)
                sim_context.collision_reports[p1] = report
        return collision
