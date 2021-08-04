from typing import List, Mapping, Tuple, MutableMapping

import numpy as np
from commonroad_dc.pycrcc import RectOBB

from games import PlayerName
from sim import ImpactLocation, CollisionReport
from sim.collision_utils import get_rectangle_mesh, get_normal_of_impact, \
    get_velocity_after_collision, get_kinetic_energy_delta, get_energy_absorbed, get_absorption_coefficient, \
    get_j_scalar_linear
from sim.simulator import SimContext


def is_a_at_fault():
    # todo this will need to be implemented with some better logic
    return False


def get_impact_locations(a_shape: RectOBB, b_shape: RectOBB) -> MutableMapping[ImpactLocation, List]:
    """
    This returns a dictionary with key: impact_location and value: [[x0,y0],[x1,y1]], points defining that vehicle
    segment
    :param a_shape: RectOBB object
    :param b_shape: RectOBB object
    :return:
    """
    locations: MutableMapping[ImpactLocation, List] = {}
    a_mesh = get_rectangle_mesh(a_shape)
    for loc, loc_shape in a_mesh.items():
        if b_shape.collide(loc_shape):
            segment = loc_shape.vertices()[0:-1]
            locations[loc] = segment
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
    a_vel_init = sim_context.models[a].get_velocity()
    b_vel_init = sim_context.models[b].get_velocity()
    # Geometry
    a_geom = sim_context.models[a].get_geometry()
    b_geom = sim_context.models[b].get_geometry()

    # Collision locations
    locations: MutableMapping[ImpactLocation, List] = get_impact_locations(a_shape, b_shape)
    abs_coefficient = get_absorption_coefficient()

    # Check if A is at fault
    at_fault: bool = is_a_at_fault()

    # Relative velocity in global RF
    rel_velocity = get_relative_velocity(a_vel_init, b_vel_init)

    # Relative velocity along normal of impact
    n = get_normal_of_impact(a_shape, b_shape)

    # Energy absorbed by passengers
    # todo if rel_velocity_along_n > 0 -> raise value error as objects would be separating
    e = min(a_geom.e, b_geom.e)  # Restitution coefficient
    j_n = get_j_scalar_linear(e, n, rel_velocity, a_geom.m, b_geom.m)
    # todo: check if next lines should be done for a or for b
    vel_final = get_velocity_after_collision(n, a_vel_init, a_geom.m, j_n)
    kenergy_delta = get_kinetic_energy_delta(a_vel_init, vel_final, a_geom.m)
    energy_passengers = get_energy_absorbed(kenergy_delta, abs_coefficient)

    return CollisionReport(location=locations,
                           at_fault=at_fault,
                           rel_velocity=rel_velocity,
                           energy_delta = kenergy_delta,
                           energy_passengers=energy_passengers,
                           at_time=sim_context.time
                           )
