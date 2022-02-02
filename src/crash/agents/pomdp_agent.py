from typing import Optional
from decimal import Decimal
from matplotlib.animation import FuncAnimation
from dg_commons import PlayerName, U
from dg_commons.sim import SimObservations, PlayerObservations
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.maps import DgLanelet
import matplotlib.pyplot as plt
from time import time
import numpy as np
from predictions import pomdp

# structure from class PID



class POMDPAgent(LFAgent):

    def __init__(self, lane: Optional[DgLanelet]):
        super().__init__(lane=lane)

        self.pomdp = "pomdp solver"


        # simulation parameters
        self.produce_animation: bool = True
        self.logging_interval: Decimal = Decimal(2)

    def on_episode_init(self, my_name: PlayerName, **kwargs):
        sim_context = kwargs['sim_context']

        #for specific example on 4 way crossing where ego starts from N and goes S
        #and there is another agent that starts from S and goes W

        # next lines hardcoded for testing purposes

        # all possible start and destination ids of the 4-way crossing
        start_ids = {'W': 1, 'E': 10,'S': 16,'N': 8} # 1: W, : 10: E, 16: S, 8: N
        destination_ids = {'W': 2,'E': 9 ,'S': 15,'N': 7} # 2: W, 9: E, 15: S, 7: N
        route_ids = {'start': start_ids, 'destination': destination_ids}

        # all routes of interest
        ego_route = [8, 13, 15]
        other_route_SW = [16, 19, 2] # route 1 #this is true route
        other_route_SE = [16, 18, 9] # route 2
        other_route_SN = [16, 14, 7] # route 3

        true_other_route = other_route_SW

        # reference velocities on these routes. Going straight allows for quicker velocities


        # here create action spaces, state domains, observation spaces

        # domains for state components
        state_v = [v_min, v_max] #continuous
        state_progress = [0,1] # continuous
        state_route = [1,2,3] # discrete

        # action domain
        allowed_accs = [-2., -1., 0., 1.]

        # observations
        # need to declare this explicitely?

        # transition (deterministic)
        A = np.array([[1., dt, 0.],[0., 1., 0.],[0., 0., 1.]])
        b = np.array([[0.5*dt*dt],[dt], [0.]])

        x_new = np.matmul(A, x_old) + b*acc








        #initial_observations = SimObservations(players={}, time=Decimal(0))
        #initial_observations.time = sim_context.time  # needed?
        #for player_name, model in sim_context.models.items():
        #    player_obs = PlayerObservations(state=model.get_state(), occupancy=model.get_footprint())
        #    initial_observations.players.update({player_name: player_obs})

        #super().on_episode_init(my_name=my_name)

        return

    def get_commands(self, sim_obs: SimObservations) -> U:
        self._my_obs = sim_obs.players[self.my_name].state
        my_pose: SE2value = SE2_from_xytheta([self._my_obs.x, self._my_obs.y, self._my_obs.theta])

        # update observations
        self.speed_behavior.update_observations(sim_obs.players)
        self.speed_controller.update_measurement(measurement=self._my_obs.vx)
        self.steer_controller.update_measurement(measurement=self._my_obs.delta)
        lanepose = self.ref_lane.lane_pose_from_SE2_generic(my_pose)
        self.pure_pursuit.update_pose(pose=my_pose, along_path=lanepose.along_lane)

        # compute commands
        t = float(sim_obs.time)
        speed_ref, emergency = self.speed_behavior.get_speed_ref(t)
        """if emergency or self._emergency:
            # Once the emergency kicks in the speed ref will always be 0
            self._emergency = True
            speed_ref = 0
            self.emergency_subroutine()"""
        self.pure_pursuit.update_speed(speed=speed_ref)
        self.speed_controller.update_reference(reference=speed_ref)
        #acc = self.speed_controller.get_control(t)
        acc = #return action from POMDP
        delta_ref = self.pure_pursuit.get_desired_steering()
        self.steer_controller.update_reference(delta_ref)
        ddelta = self.steer_controller.get_control(t)
        return VehicleCommands(acc=acc, ddelta=ddelta, lights=self.lights_test_seq.at_or_previous(sim_obs.time))


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
