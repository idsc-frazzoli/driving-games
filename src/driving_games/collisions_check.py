from itertools import combinations
from typing import Mapping

from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from dg_commons import PlayerName
from dg_commons.sim import CollisionReportPlayer
from dg_commons.sim.collision_utils import (
    chek_who_is_at_fault,
    compute_impact_geometry,
    velocity_after_collision,
    rot_velocity_after_collision,
    kinetic_energy,
    compute_impulse_response,
)
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from games import GameConstants
from .structures import VehicleState

__all__ = ["collision_check"]


# todo: Note that this only works for the simplest cases.
#      For example it does not work for head-to-back collision.
def collision_check(
    poses: Mapping[PlayerName, VehicleState],
    # todo probably here it would be better to simply pass a Trajectory for the interval previous state and current
    geometries: Mapping[PlayerName, VehicleGeometry],
) -> Mapping[PlayerName, CollisionReportPlayer]:
    if GameConstants.checks:
        assert set(poses.keys()) == set(geometries.keys())

    for p1, p2 in combinations(poses, 2):
        s1, s2 = poses[p1], poses[p2]
        g1, g2 = geometries[p1], geometries[p2]

        a_shape: Polygon = g1.outline_as_polygon
        b_shape: Polygon = g2.outline_as_polygon

        # dt = D("0.5") #todo should sample between current state and previous one
        # n = 2
        # x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
        # x2s = sample_x(s2.x, s2.v, dt=dt, n=n)

        for x1, x2 in zip(
            [
                s1,
            ],
            [
                s2,
            ],
        ):
            a_matrix_coeff = x1[0, :2].tolist() + x1[1, :2].tolist() + x1[:2, 2].tolist()
            a_shape_tra = affine_transform(a_shape, a_matrix_coeff)
            b_matrix_coeff = x2[0, :2].tolist() + x2[1, :2].tolist() + x2[:2, 2].tolist()
            b_shape_tra = affine_transform(b_shape, b_matrix_coeff)
            if a_shape_tra.intersects(b_shape_tra):  # collision
                impact_normal, impact_point = compute_impact_geometry(a_shape, b_shape)
                p_at_fault = chek_who_is_at_fault()
                j_n = compute_impulse_response(
                    n=impact_normal, vel_ab=rel_velocity_atP, r_ap=r_ap, r_bp=r_bp, a_geom=g1, b_geom=g2
                )
                # Apply impulse to models
                a_vel_after = velocity_after_collision(impact_normal, a_vel, g1.m, j_n)
                b_vel_after = velocity_after_collision(-impact_normal, b_vel, g2.m, j_n)
                a_omega_after = rot_velocity_after_collision(r_ap, impact_normal, a_omega, g1.Iz, j_n)
                b_omega_after = rot_velocity_after_collision(r_bp, -impact_normal, b_omega, g2.Iz, j_n)

                # Log reports
                a_kenergy_delta = kinetic_energy(a_vel_after, g1.m) - kinetic_energy(a_vel, g1.m)
                b_kenergy_delta = kinetic_energy(b_vel_after, g2.m) - kinetic_energy(b_vel, g2.m)
                # todo rotational energy
                a_report = CollisionReportPlayer(
                    locations=a_locations,
                    at_fault=p_at_fault[p1],
                    footprint=a_shape,
                    velocity=(a_vel, a_omega),
                    velocity_after=(a_vel_after, a_omega_after),
                    energy_delta=a_kenergy_delta,
                )
                b_report = CollisionReportPlayer(
                    locations=b_locations,
                    at_fault=p_at_fault[p2],
                    footprint=b_shape,
                    velocity=(b_vel, b_omega),
                    velocity_after=(b_vel_after, b_omega_after),
                    energy_delta=b_kenergy_delta,
                )
                return {p1: a_report, p2: b_report}
    return {}
