from typing import Any, Optional, Dict, Mapping, List
from decimal import Decimal
from itertools import product

from dg_commons import PlayerName, U, X
from dg_commons.sim.agents.lane_follower import LFAgent
from predictions.goals import GoalGenerator
from dg_commons.sim.simulator import SimContext

class PredAgent(LFAgent):

    #__init__ and on_get_extra copy-pasted from B2Agent. Implementation not optimal. todo: make more elegant
    #pred_module: Mapping[PlayerName, X] = {}  # todo: extend from point prediction to entire prediction or probability
    goal_generator: GoalGenerator
    #todo: current workaround to give goal as trajectory needs to be fixed in simulator, i.e. simulator should also be able to handle other types of extra
    def on_get_extra(self, sim_context: SimContext) -> List[X]:
        goals = self.goal_generator.infer_goals(sim_context) # the following commented lines should be in goal_generator
        candidates = tuple(product(goals,["gold", ],))
        return candidates

'''
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

'''    def on_get_extra(
        self,
    ) -> [OptionalDrawableTrajectoryType]: #may change trajectory to full "sets"
        return self.pred_module.dump_drawable()'''
