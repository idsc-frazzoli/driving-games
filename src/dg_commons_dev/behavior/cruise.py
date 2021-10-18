from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
from dg_commons_dev.behavior.utils import SituationObservations
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from math import pi


@dataclass
class CruiseSituation:
    is_cruise: bool = True

    speed_ref: Optional[float] = None

    def __post_init__(self):
        if self.is_cruise:
            assert self.speed_ref is not None


@dataclass
class CruiseParams(SituationParams):
    nominal_speed: Union[List[float], float] = kmh2ms(40)
    """Nominal desired speed"""


class Cruise(Situation[SituationObservations, CruiseSituation]):
    def __init__(self, params: CruiseParams, safety_time_braking):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.obs: Optional[SituationObservations] = None
        self.cruise_situation: Optional[CruiseSituation] = None

    def update_observations(self, new_obs: SituationObservations):
        self.obs = new_obs
        my_name = new_obs.my_name
        agents = new_obs.agents
        agents_rel_pose = new_obs.rel_poses

        my_vel = agents[my_name].state.vx
        candidate_speed_ref = [self.params.nominal_speed, ]
        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            rel = agents_rel_pose[other_name]
            rel_dist = np.linalg.norm(rel.p)
            other_vel = agents[other_name].state.vx
            in_front_of_me: bool = rel.p[0] > 0.5 and abs(rel.p[1]) <= 1.2 and abs(rel.theta) < pi / 6
            # safety distance at current speed + difference of how it will be in the next second
            dist_to_keep = self._get_min_safety_dist(my_vel) + max(my_vel - other_vel, 0)
            if in_front_of_me and rel_dist < dist_to_keep:
                speed_ref = float(np.clip(other_vel - max(my_vel - other_vel, 0), 0, kmh2ms(130)))
                candidate_speed_ref.append(speed_ref)
            else:
                candidate_speed_ref.append(self.params.nominal_speed)

        self.cruise_situation = CruiseSituation(speed_ref=min(candidate_speed_ref))

    def _get_min_safety_dist(self, vel: float):
        """The distance covered in x [s] travelling at vel"""
        return vel * self.safety_time_braking

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.cruise_situation.is_cruise

    def infos(self) -> CruiseSituation:
        assert self.obs is not None
        return self.cruise_situation
