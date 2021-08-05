from typing import List

from shapely.geometry import Polygon

from games import PlayerName
from sim import ImpactLocation, CollisionReport
from sim.collision_structures import CollisionReportPlayer
from sim.collision_utils import get_rectangle_mesh, compute_impact_geometry, \
    get_velocity_after_collision, kinetic_energy, compute_impulse_response
from sim.simulator import SimContext


def is_a_at_fault():
    # todo this will need to be implemented with some better logic
    return False


def impact_locations_from_polygons(a_shape: Polygon, b_shape: Polygon) -> List[ImpactLocation]:
    """
    This returns a dictionary with key: impact_location and value: [[x0,y0],[x1,y1]], points defining that vehicle
    segment
    :param a_shape: RectOBB object
    :param b_shape: RectOBB object
    :return:
    """
    locations: List[ImpactLocation] = []
    a_mesh = get_rectangle_mesh(a_shape)
    for loc, loc_shape in a_mesh.items():
        if b_shape.intersects(loc_shape):
            locations.append(loc)
    if not locations:
        raise RuntimeWarning("Detected a collision but unable to find the impact location")
    return locations


def compute_collision_report(a: PlayerName, b: PlayerName, sim_context: SimContext) -> CollisionReport:
    """
    Compute the collision report for A given that A and B collided
    """
    # Compute collision locations
    a_shape = sim_context.models[a].get_footprint()
    b_shape = sim_context.models[b].get_footprint()
    a_locations = impact_locations_from_polygons(a_shape, b_shape)
    b_locations = impact_locations_from_polygons(b_shape, a_shape)
    # Check if A is at fault
    a_fault: bool = is_a_at_fault()
    b_fault: bool = is_a_at_fault()
    # todo check based on velocities if we actually need to resolve the collision
    # Velocity in global RF
    a_vel_init = sim_context.models[a].get_velocity()
    b_vel_init = sim_context.models[b].get_velocity()
    # Geometry
    a_geom = sim_context.models[a].get_geometry()
    b_geom = sim_context.models[b].get_geometry()
    # Relative velocity in global RF
    rel_velocity = a_vel_init - b_vel_init
    # Relative velocity along normal of impact
    impact_normal, impact_point = compute_impact_geometry(a_shape, b_shape)
    # Energy absorbed by passengers
    # todo if rel_velocity_along_n > 0 -> raise value error as objects would be separating
    r_ap = impact_point - a_shape.centroid
    r_bp = impact_point - b_shape.centroid
    j_n = compute_impulse_response(vec=impact_normal,
                                   rel_v=rel_velocity,
                                   r_ap=r_ap,
                                   r_bp=r_bp,
                                   a_geom=a_geom,
                                   b_geom=b_geom)
    # todo: check if next lines should be done for a or for b
    a_vel_final = get_velocity_after_collision(impact_normal, a_vel_init, a_geom.m, j_n)
    b_vel_final = get_velocity_after_collision(impact_normal, b_vel_init, b_geom.m, j_n)
    a_kenergy_delta = kinetic_energy(a_vel_final, a_geom.m) - kinetic_energy(a_vel_init, a_geom.m)
    b_kenergy_delta = kinetic_energy(b_vel_final, b_geom.m) - kinetic_energy(b_vel_init, b_geom.m)
    # todo assorbtion coefficient needs to be justified

    a_report = CollisionReportPlayer(locations=a_locations,
                                     at_fault=a_fault,
                                     footprint=a_shape,
                                     velocity=a_vel_init,
                                     energy_delta=a_kenergy_delta, )
    b_report = CollisionReportPlayer(locations=b_locations,
                                     at_fault=b_fault,
                                     footprint=b_shape,
                                     velocity=b_vel_init,
                                     energy_delta=b_kenergy_delta, )
    return CollisionReport(players={a: a_report, b: b_report},
                           impact_point=impact_point,
                           impact_normal=impact_normal,
                           at_time=sim_context.time,
                           )
