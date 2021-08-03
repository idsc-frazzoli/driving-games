from typing import List

import numpy as np
from commonroad_dc.pycrcc import RectOBB

from games import PlayerName
from sim import ImpactLocation, CollisionReport
from sim.collision_utils import get_rectangle_mesh
from sim.simulator import SimContext


def is_a_at_fault():
    # todo this will need to be implemented with some better logic
    return False


def get_impact_locations(a_shape: RectOBB, b_shape: RectOBB) -> List[ImpactLocation]:
    locations: List[ImpactLocation] = []
    a_mesh = get_rectangle_mesh(a_shape)
    for loc, loc_shape in a_mesh.items():
        if b_shape.collide(loc_shape):
            locations.append(loc)
    if not locations:
        raise RuntimeWarning("Detected a collision but unable to find the impact location")
    return locations


def get_relative_velocity(a_vel: np.ndarray, b_vel: np.ndarray) -> np.ndarray:
    return a_vel - b_vel


def compute_collision_report(a: PlayerName, b: PlayerName, sim_context: SimContext) -> CollisionReport:
    """
    Compute the collision report for A given that A and B collided
    """

    # Footprints
    a_shape = sim_context.models[a].get_footprint()
    b_shape = sim_context.models[b].get_footprint()
    # Velocity in global RF
    a_vel = sim_context.models[a].get_velocity()
    b_vel = sim_context.models[b].get_velocity()

    # First we get the collision locations
    locations: List[ImpactLocation] = get_impact_locations(a_shape, b_shape)
    # Check if A is at fault
    at_fault: bool = is_a_at_fault()
    # todo relative velocity
    # rel_velocity = np.array([0, 0])
    rel_velocity = get_relative_velocity(a_vel, b_vel)
    # todo relative velocity along normal
    rel_velocity_along_n = 0
    # todo energy transferred
    energy = 0
    return CollisionReport(location=locations,
                           at_fault=at_fault,
                           rel_velocity=rel_velocity,
                           rel_velocity_along_n = rel_velocity_along_n,
                           energy_transfer=energy,
                           at_time=sim_context.time
                           )
