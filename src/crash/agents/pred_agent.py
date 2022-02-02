from typing import Any, List, Dict, Optional
from decimal import Decimal
from matplotlib.animation import FuncAnimation
import numpy as np
from dg_commons.sim.simulator import SimContext
from dg_commons import PlayerName, U
# from dg_commons.maps.road_networks import DynamicRoadGraph, get_collections_networkx_temp
from dg_commons.maps.road_networks_tmp import DynamicRoadGraph, get_collections_networkx_temp
from dg_commons.sim import SimObservations, PlayerObservations
from dg_commons.sim.agents.lane_follower import LFAgent
from commonroad.scenario.scenario import LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblem
from dg_commons.maps import DgLanelet
import matplotlib.pyplot as plt
from time import time


class PredAgent(LFAgent):

    def __init__(self, lane: Optional[DgLanelet], lanelet_network: LaneletNetwork,
                 planning_problem: PlanningProblem, max_length: Optional[float] = None):
        super().__init__(lane=lane)
        # instantiate dynamic graph for predictions
        self.dynamic_graph: DynamicRoadGraph = DynamicRoadGraph(lanelet_network=lanelet_network, max_length=max_length)

        # compute start and goal nodes for ego agent
        self.dynamic_graph.start_and_goal_info(problem=planning_problem)

        # simulation parameters
        self.produce_animation: bool = True
        self.logging_interval: Decimal = Decimal(2)

    def on_episode_init(self, my_name: PlayerName, **kwargs):
        sim_context = kwargs['sim_context']
        initial_observations = SimObservations(players={}, time=Decimal(0))
        initial_observations.time = sim_context.time  # needed?
        for player_name, model in sim_context.models.items():
            player_obs = PlayerObservations(state=model.get_state(), occupancy=model.get_footprint())
            initial_observations.players.update({player_name: player_obs})

        # initialize prediction object and player locations based on initial observations.
        self.dynamic_graph.initialize_prediction(initial_obs=initial_observations)
        super().on_episode_init(my_name=my_name)
        return

    def get_commands(self, sim_obs: SimObservations) -> U:
        t1 = time()

        # update all predictions with new observations
        self.dynamic_graph.update_predictions(sim_obs=sim_obs)

        # printing predictions to terminal
        if sim_obs.time % self.logging_interval == 0:
            print("Current sim time is: " + str(sim_obs.time))
            print("Probabilities predicted: ")
            print(self.dynamic_graph.prediction.probabilities)
            print("Duration of prediction update loop: " + str(time() - t1))

        if self.produce_animation:
            # store digraph at each timestamp
            self.dynamic_graph.keep_track()

        return super().get_commands(sim_obs)

    # only for animation generation
    def on_episode_end(self):
        if self.produce_animation:
            fig, ax = plt.subplots(figsize=(30, 30))
            colls = self.dynamic_graph.graph_storage

            def animate(i):
                print(i)
                graph = colls[i]
                _, _, _ = get_collections_networkx_temp(resource_graph=graph, ax=ax)

            print("Number of frames: " + str(len(colls)))
            frames = int(len(colls) / 10) + 1
            print("Frames: " + str(frames))
            anim = FuncAnimation(fig, animate, interval=1000, frames=frames)
            dpi = 120
            t1 = time()
            anim.save('basic_test.gif', dpi=dpi, writer="ffmpeg")
            print("Time needed to produce animation fro pre-stored graph states: " + str(time() - t1))
