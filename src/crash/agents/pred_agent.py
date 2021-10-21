from typing import Any, Optional, Dict
from decimal import Decimal
from itertools import product

from dg_commons import PlayerName, U, X
from dg_commons.planning.motion_primitives import MPGParam, MotionPrimitivesGenerator
from dg_commons.dynamics import BicycleDynamics
from crash.agents.b2agent import B2Agent
from sim.agents.lane_follower import LFAgent
from sim.models.vehicle_structures import VehicleGeometry
from sim.models.vehicle_utils import VehicleParameters
from sim import SimObservations
from sim import DrawableTrajectoryType

class PredAgent(LFAgent):

#__init__ and on_get_extra copy-pasted from B2Agent. Implementation not optimal. todo: make more elegant
    pred_module: Dict[PlayerName, X] = {}  # todo: extend from point prediction to entire prediction or probability

'''    def on_get_extra(
        self,
    ) -> Optional[DrawableTrajectoryType]:
        trajectories = self._mpg.generate(x0=self._my_obs.to_vehicle_state())
        if len(trajectories) == 0:
            return None
        candidates = tuple(
            product(
                trajectories,
                [
                    "gold",
                ],
            )
        )
        return candidates'''




    def get_commands(self, sim_obs: SimObservations) -> U:
        # todo predictions from obsrv: Work in progress
        for player in sim_obs.players:
            if player.lower() == 'ego':
                continue
            self.pred_module[player] = sim_obs.players[player].state
            self.pred_module[player].x = self.pred_module[player].x + 2
            self.pred_module[player].y = self.pred_module[player].y + 1

        # todo
        # 1) get all observations from other participants -> aöready done in "update" by calling this method wit sim_obs as input
        # 2) assume constant velocity
        # 3) predict trajectories for all other participants for m seconds
        # 4) draw trajectories on simulation
        # 5) repeat every n microseconds


        # todo: questo ritorna ciò che viene fatto nella get_commands() di LFAgent?
        return super().get_commands(sim_obs)
    '''
    def on_get_extra(
        self,
    ) -> [OptionalDrawableTrajectoryType]: #may change trajectory to full "sets"
        return self.pred_module.dump_drawable()'''
