from dataclasses import dataclass
from math import pi
from typing import MutableMapping, Dict, Optional, List, Union, Tuple
import numpy as np
from duckietown_world import relative_pose, SE2Transform
from geometry import SE2value
from dg_commons import PlayerName
from games.utils import valmap
from dg_commons.sim.models import extract_pose_from_state, kmh2ms, extract_vel_from_state
from dg_commons.sim.simulator_structures import PlayerObservations
from dg_commons_dev.behavior.behavior_types import Behavior, BehaviorParams, Situation, SituationParams
from dg_commons_dev.behavior.emergency import EmergencySituation, Emergency, EmergencyParams
from dg_commons_dev.behavior.yield_to import YieldSituation, Yield, YieldParams
from dg_commons_dev.behavior.cruise import CruiseSituation, CruiseParams, Cruise
from dg_commons_dev.behavior.utils import SituationObservations


@dataclass
class BehaviorSituation:
    yield_to: Optional[YieldSituation] = None
    emergency: Optional[EmergencySituation] = None
    cruise: Optional[CruiseSituation] = None


@dataclass
class SpeedBehaviorParam(BehaviorParams):
    safety_time_braking: Union[List[float], float] = 1.5
    """Evaluates safety distance from vehicle in front based on distance covered in this delta time"""
    emergency: type(Emergency) = Emergency
    emergency_params: EmergencyParams = EmergencyParams()
    """ Emergency Behavior """
    yield_to: type(Emergency) = Yield
    yield_params: YieldParams = YieldParams()
    """ Yield Behavior """
    cruise: type(Cruise) = Cruise
    cruise_params: CruiseParams = CruiseParams()
    """ Cruise Params """


class SpeedBehavior(Behavior[MutableMapping[PlayerName, PlayerObservations], Tuple[float, BehaviorSituation]]):
    """Determines the reference speed"""

    def __init__(self, params: SpeedBehaviorParam = SpeedBehaviorParam(), my_name: Optional[PlayerName] = None):
        self.params: SpeedBehaviorParam = params
        self.my_name: PlayerName = my_name
        self.agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None
        self.speed_ref: float = 0

        self.yield_to = self.params.yield_to(self.params.yield_params, self.params.safety_time_braking)
        self.emergency = self.params.emergency(self.params.emergency_params, self.params.safety_time_braking)
        self.cruise = self.params.cruise(self.params.cruise_params, self.params.safety_time_braking)
        self.obs: SituationObservations = SituationObservations(my_name=self.my_name)
        self.situation: BehaviorSituation = BehaviorSituation()
        """ The speed reference"""

    def update_observations(self, agents: MutableMapping[PlayerName, PlayerObservations]):
        self.agents = agents
        self.obs.agents = agents

    def get_situation(self, at: float) -> Tuple[float, BehaviorSituation]:
        self.obs.my_name = self.my_name
        mypose = extract_pose_from_state(self.agents[self.my_name].state)

        def rel_pose(other_obs: PlayerObservations) -> SE2Transform:
            other_pose: SE2value = extract_pose_from_state(other_obs.state)
            return SE2Transform.from_SE2(relative_pose(mypose, other_pose))

        agents_rel_pose: Dict[PlayerName, SE2Transform] = valmap(rel_pose, self.agents)
        self.obs.rel_poses = agents_rel_pose
        self.yield_to.update_observations(self.obs)
        self.emergency.update_observations(self.obs)
        self.situation.yield_to = self.yield_to.infos()
        self.situation.emergency = self.emergency.infos()

        if self.situation.yield_to.is_yield or self.situation.emergency.is_emergency:
            self.situation.cruise = CruiseSituation(False)
            self.speed_ref = 0
        else:
            self.cruise.update_observations(self.obs)
            self.situation.cruise = self.cruise.infos()
            self.speed_ref = self.cruise.infos().speed_ref

        return self.speed_ref, self.situation

    def is_there_anyone_to_yield_to(self) -> bool:
        return self.yield_to.is_true()

    def is_emergency_subroutine_needed(self) -> bool:
        return self.emergency.is_true()

    def is_cruise(self) -> bool:
        return self.cruise.is_true()