from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from games import PlayerName
from sim import ImpactLocation, CollisionReport, logger
from sim.collision_structures import CollisionReportPlayer
from sim.collision_utils import get_rectangle_mesh, compute_impact_geometry, \
    velocity_after_collision, kinetic_energy, compute_impulse_response, rot_velocity_after_collision
from sim.simulator import SimContext


def is_a_at_fault():
    # todo this will need to be implemented with some better logic
    return False


def impact_locations_from_polygons(a_shape: Polygon, b_shape: Polygon) -> List[Tuple[ImpactLocation, Polygon]]:
    """
    This returns a dictionary with key: impact_location and value: [[x0,y0],[x1,y1]], points defining that vehicle
    segment
    :param a_shape: RectOBB object
    :param b_shape: RectOBB object
    :return:
    """
    locations: List[Tuple[ImpactLocation, Polygon]] = []
    a_mesh = get_rectangle_mesh(a_shape)
    for loc, loc_shape in a_mesh.items():
        if b_shape.intersects(loc_shape):
            locations.append((loc, loc_shape))
    if not locations:
        raise RuntimeWarning("Detected a collision but unable to find the impact location")
    return locations


def resolve_collision(a: PlayerName, b: PlayerName, sim_context: SimContext) -> Optional[CollisionReport]:
    """
    Resolves the collision between A and B using an impulse method.
    :returns A CollisionReport or None if the collision does not need to be solved (the two bodies are already separating)
    """
    a_shape = sim_context.models[a].get_footprint()
    b_shape = sim_context.models[b].get_footprint()
    # Compute collision geometry
    impact_normal, impact_point = compute_impact_geometry(a_shape, b_shape)
    a_vel, a_omega = sim_context.models[a].get_velocity(in_model_frame=False)
    b_vel, b_omega = sim_context.models[b].get_velocity(in_model_frame=False)
    rel_velocity = a_vel - b_vel
    if np.dot(rel_velocity, impact_normal) < 0:
        logger.debug(f"Not solving the collision between {a}, {b} since they are already separating")
        return None

    # Compute collision locations
    a_locations = impact_locations_from_polygons(a_shape, b_shape)
    b_locations = impact_locations_from_polygons(b_shape, a_shape)

    # Check who is at fault
    a_fault: bool = is_a_at_fault()
    b_fault: bool = is_a_at_fault()

    # Compute impulse resolution
    a_geom = sim_context.models[a].get_geometry()
    b_geom = sim_context.models[b].get_geometry()
    r_ap = np.array(impact_point.coords[0]) - np.array(a_shape.centroid.coords[0])
    r_bp = np.array(impact_point.coords[0]) - np.array(b_shape.centroid.coords[0])
    j_n = compute_impulse_response(n=impact_normal,
                                   vel_ab=rel_velocity,
                                   r_ap=r_ap,
                                   r_bp=r_bp,
                                   a_geom=a_geom,
                                   b_geom=b_geom)
    # Apply impulse to models
    a_vel_after = velocity_after_collision(impact_normal, a_vel, a_geom.m, j_n)
    b_vel_after = velocity_after_collision(-impact_normal, b_vel, b_geom.m, j_n)
    a_omega_after = rot_velocity_after_collision(r_ap, impact_normal, a_omega, a_geom.Iz, j_n)
    b_omega_after = rot_velocity_after_collision(r_bp, -impact_normal, b_omega, b_geom.Iz, j_n)
    sim_context.models[a].set_velocity(a_vel_after, a_omega_after, in_model_frame=False)
    sim_context.models[b].set_velocity(b_vel_after, b_omega_after, in_model_frame=False)

    # Log reports
    a_kenergy_delta = kinetic_energy(a_vel_after, a_geom.m) - kinetic_energy(a_vel, a_geom.m)
    b_kenergy_delta = kinetic_energy(b_vel_after, b_geom.m) - kinetic_energy(b_vel, b_geom.m)
    # todo rotational energy
    a_report = CollisionReportPlayer(locations=a_locations,
                                     at_fault=a_fault,
                                     footprint=a_shape,
                                     velocity=(a_vel, a_omega),
                                     velocity_after=(a_vel_after, a_omega_after),
                                     energy_delta=a_kenergy_delta)
    b_report = CollisionReportPlayer(locations=b_locations,
                                     at_fault=b_fault,
                                     footprint=b_shape,
                                     velocity=(b_vel, b_omega),
                                     velocity_after=(b_vel_after, b_omega_after),
                                     energy_delta=b_kenergy_delta)
    return CollisionReport(players={a: a_report, b: b_report},
                           impact_point=impact_point,
                           impact_normal=impact_normal,
                           at_time=sim_context.time,
                           )
