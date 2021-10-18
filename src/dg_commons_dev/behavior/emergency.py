from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
from dg_commons_dev.behavior.utils import SituationObservations
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from math import pi


@dataclass
class EmergencySituation:
    is_emergency: bool

    def __post_init__(self):
        if self.is_emergency:
            pass


@dataclass
class EmergencyParams(SituationParams):
    min_dist: Union[List[float], float] = 7
    """Evaluate emergency only for vehicles within x [m]"""
    min_vel: Union[List[float], float] = kmh2ms(5)
    """emergency only to vehicles that are at least moving at.."""


class Emergency(Situation[SituationObservations, EmergencySituation]):
    def __init__(self, params: EmergencyParams, safety_time_braking):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.obs: Optional[SituationObservations] = None
        self.emergency_situation: Optional[EmergencySituation] = None

    def update_observations(self, new_obs: SituationObservations):
        self.obs = new_obs
        my_name = new_obs.my_name
        agents = new_obs.agents
        agents_rel_pose = new_obs.rel_poses

        my_vel = agents[my_name].state.vx
        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            rel = agents_rel_pose[other_name]
            other_vel = extract_vel_from_state(agents[other_name].state)
            rel_distance = np.linalg.norm(rel.p)
            coming_from_the_left: bool = -3 * pi / 4 <= rel.theta <= -pi / 4 and \
                other_vel > self.params.min_vel
            in_front_of_me: bool = rel.p[0] > 0 and - 1.2 <= rel.p[1] <= 1.2
            coming_from_the_front: bool = 3 * pi / 4 <= abs(rel.theta) <= pi * 5 / 4 and in_front_of_me
            if (coming_from_the_left and rel_distance < self.params.min_dist) or (
                    coming_from_the_front and rel_distance < self.params.safety_time_braking * (my_vel + other_vel)):
                self.emergency_situation = EmergencySituation(True)
        self.emergency_situation = EmergencySituation(False)

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.emergency_situation.is_emergency

    def infos(self) -> EmergencySituation:
        assert self.obs is not None
        return self.emergency_situation
