from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Mapping, Optional, List, MutableMapping

from commonroad.scenario.scenario import Scenario

from dg_commons.time import time_function
from games import PlayerName, U
from sim import logger, CollisionReport, SimTime
from sim.agents.agent import Agent
from sim.collision_utils import CollisionException
from sim.simulator_structures import *


@dataclass
class SimContext:
    """ The simulation context that keeps track of everything, handle with care as it is passed around by reference and
    it is a mutable object"""
    scenario: Scenario
    models: Mapping[PlayerName, SimModel]
    players: Mapping[PlayerName, Agent]
    log: SimulationLog
    param: SimParameters
    time: SimTime = Decimal(0)
    seed: int = 0
    sim_terminated: bool = False
    collision_reports: List[CollisionReport] = field(default_factory=list)
    first_collision_ts: SimTime = Decimal(999)

    def __post_init__(self):
        assert self.models.keys() == self.players.keys()
        assert isinstance(self.scenario, Scenario), self.scenario
        for pname in self.models.keys():
            assert issubclass(type(self.models[pname]), SimModel)
            assert issubclass(type(self.players[pname]), Agent)


class Simulator:
    last_observations: Optional[SimObservations] = SimObservations(players={}, time=Decimal(0))
    last_get_commands_ts: SimTime = SimTime(-99)
    last_commands: MutableMapping[PlayerName, U] = {}

    @time_function
    def run(self, sim_context: SimContext):
        logger.info("Beginning simulation.")
        for player_name, player in sim_context.players.items():
            player.on_episode_init(player_name)
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)
        logger.info("Completed simulation.")

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
        update_commands: bool = (sim_context.time - self.last_get_commands_ts) >= sim_context.param.dt_commands
        # fixme this can be parallelized later with ProcessPoolExecutor?
        for player_name, model in sim_context.models.items():
            if update_commands:
                actions = sim_context.players[player_name].get_commands(self.last_observations)
                self.last_commands[player_name] = actions
            cmds = self.last_commands[player_name]
            model.update(cmds, dt=sim_context.param.dt)
            log_entry = LogEntry(state=model.get_state(), actions=cmds)
            logger.debug(f"Update function, sim time {sim_context.time:.2f}, player: {player_name}")
            logger.debug(f"New state {model.get_state()} reached applying {cmds}")
            sim_context.log[sim_context.time].update({player_name: log_entry})
        if update_commands:
            self.last_get_commands_ts = sim_context.time
        return

    def post_update(self, sim_context: SimContext):
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        :param sim_context:
        :return:
        """
        collision_detected = self._check_collisions(sim_context)
        # after all the computations advance simulation time
        sim_context.time += sim_context.param.dt
        self._maybe_terminate_simulation(sim_context)
        return

    @staticmethod
    def _maybe_terminate_simulation(sim_context: SimContext):
        """ Evaluates if the simulation needs to terminate based on the expiration of times"""
        termination_condition: bool = \
            sim_context.time > sim_context.param.max_sim_time or \
            sim_context.time > sim_context.first_collision_ts + sim_context.param.sim_time_after_collision
        sim_context.sim_terminated = termination_condition

    @staticmethod
    def _check_collisions(sim_context: SimContext) -> bool:
        """
        This checks only collision location at the current step, tunneling effects and similar are ignored
        :param sim_context:
        :return: True if at least one collision happened, False otherwise
        """
        collision = False
        for p1, p2 in combinations(sim_context.models, 2):
            a_shape = sim_context.models[p1].get_footprint()
            b_shape = sim_context.models[p2].get_footprint()
            if a_shape.intersects(b_shape):
                from sim.collision import resolve_collision  # import here to avoid circular imports
                try:
                    report: Optional[CollisionReport] = resolve_collision(p1, p2, sim_context)
                except CollisionException as e:
                    logger.warn(f"Failed to resolve collision between {p1} and {p2} because:\n{e.args}")
                    report = None
                if report is not None:
                    logger.info(f"Detected a collision between {p1} and {p2}")
                    collision = True
                    if report.at_time < sim_context.first_collision_ts:
                        sim_context.first_collision_ts = report.at_time
                    sim_context.collision_reports.append(report)
        return collision
