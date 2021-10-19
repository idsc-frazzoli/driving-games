from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
from dg_commons_dev.behavior.utils import SituationObservations, front_polygon, relative_velocity
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from math import pi
from shapely.geometry import Polygon
from dg_commons.sim.models.vehicle import VehicleParameters


def drac_f():
    pass


def ttc_f():
    pass


def pet_f():
    pass


def psd_f():
    pass


@dataclass
class EmergencySituation:
    is_emergency: bool = False

    drac: Optional[float] = None
    ttc: Optional[float] = None
    pet: Optional[float] = None

    def __post_init__(self):
        if self.is_emergency:
            assert self.ttc is not None
            assert self.drac is not None


@dataclass
class EmergencyParams(SituationParams):
    min_dist: Union[List[float], float] = 7
    """Evaluate emergency only for vehicles within x [m]"""
    min_vel: Union[List[float], float] = kmh2ms(5)
    """emergency only to vehicles that are at least moving at.."""


class Emergency(Situation[SituationObservations, EmergencySituation]):
    def __init__(self, params: EmergencyParams, safety_time_braking,
                 vehicle_params: VehicleParameters = VehicleParameters.default_car()):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.obs: Optional[SituationObservations] = None
        self.emergency_situation: EmergencySituation = EmergencySituation()
        self.acc_limits = vehicle_params.acc_limits

    def update_observations(self, new_obs: SituationObservations):
        self.obs = new_obs
        my_name = new_obs.my_name
        agents = new_obs.agents
        agents_rel_pose = new_obs.rel_poses

        my_vel = agents[my_name].state.vx
        my_occupancy = agents[my_name].occupancy
        my_polygon = front_polygon(my_occupancy, self._get_min_safety_dist(my_vel))
        follow_lead = False

        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            other_vel = extract_vel_from_state(agents[other_name].state)
            rel_pose = agents_rel_pose[other_name].as_SE2()
            rel_velocity = relative_velocity(my_vel, other_vel, rel_pose)
            other_occupancy = agents[other_name].occupancy
            rel_distance = my_occupancy.distance(other_occupancy)
            other_polygon = front_polygon(other_occupancy, self._get_min_safety_dist(other_vel))

            if follow_lead:
                ttc = None if rel_velocity <= 0 else rel_distance / rel_velocity
                drac = 0.5 * rel_velocity ** 2 / rel_distance
                pet = None
                self.emergency_situation.ttc = ttc
                self.emergency_situation.drac = drac
                self.emergency_situation.pet = pet
                if drac > 0.7:
                    self.emergency_situation.is_emergency = True
            else:
                intersection = my_polygon.intersection(other_polygon)
                if intersection.is_empty:
                    self.emergency_situation = EmergencySituation(False)
                else:
                    length = intersection.length
                    my_vel = my_vel if my_vel != 0 else 10e-6
                    other_vel = other_vel if other_vel != 0 else 10e-6

                    my_distance, other_distance = my_occupancy.distance(intersection), \
                        other_occupancy.distance(intersection)
                    my_entry_time, other_entry_time = my_distance/my_vel, other_distance/other_vel
                    my_delta, other_delta = length/my_vel, length/other_vel
                    my_exit_time, other_exit_time = my_entry_time + my_delta, other_entry_time + other_delta

                    pot1, pot2 = my_exit_time - other_entry_time > 0, other_exit_time - my_entry_time > 0
                    if pot1 or pot2:
                        ttc = other_distance / other_vel if my_entry_time < other_entry_time else my_distance / my_vel
                        drac = 2 * (other_vel - other_distance/my_exit_time)/my_exit_time if pot1 else \
                            2 * (my_vel - my_distance / other_exit_time) / other_exit_time
                        pet = my_exit_time - other_entry_time if pot1 else other_exit_time - my_entry_time
                        self.emergency_situation.ttc = ttc
                        self.emergency_situation.drac = drac
                        self.emergency_situation.pet = pet
                        if drac > self.acc_limits[1] or ttc < self.safety_time_braking:
                            self.emergency_situation.is_emergency = True
                            print("yes")

    def _get_min_safety_dist(self, vel: float):
        """The distance covered in x [s] travelling at vel"""
        return vel * self.safety_time_braking

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.emergency_situation.is_emergency

    def infos(self) -> EmergencySituation:
        assert self.obs is not None
        return self.emergency_situation
