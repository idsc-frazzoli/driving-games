from itertools import combinations
from math import pi
from typing import Mapping, Dict

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from geometry import T2value, SO2value, SO2_from_angle
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, Point
from zuper_commons.types import ZValueError

from dg_commons import PlayerName, DgSampledSequence, Timestamp
from dg_commons.sim import CollisionReportPlayer, ImpactLocation, IMPACT_FRONT, IMPACT_LEFT, IMPACT_BACK, IMPACT_RIGHT
from dg_commons.sim.collision_utils import (
    check_who_is_at_fault,
    compute_impact_geometry,
    velocity_after_collision,
    rot_velocity_after_collision,
    kinetic_energy,
    compute_impulse_response,
    get_impact_point_direction,
)
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from driving_games import VehicleJointCost
from games import GameConstants

__all__ = ["joint_collision_check"]


def joint_collision_check(
    transitions: Mapping[PlayerName, DgSampledSequence[VehicleState]],
    geometries: Mapping[PlayerName, VehicleGeometry],
    col_dt: Timestamp,
    lanelet_network: LaneletNetwork,
    min_safety_dist: float,
) -> Mapping[PlayerName, VehicleJointCost]:
    """This is an involved version of the collision check."""
    if GameConstants.checks:
        assert set(transitions.keys()) == set(geometries.keys())
    accidents: Dict[PlayerName, VehicleJointCost] = {}

    for player1, player2 in combinations(transitions, 2):
        trans1, trans2 = transitions[player1], transitions[player2]
        g1, g2 = geometries[player1], geometries[player2]

        a_shape: Polygon = g1.outline_as_polygon
        b_shape: Polygon = g2.outline_as_polygon

        # we up-sample the transition according to col_dt from the end going backwards
        t1_end, t1_start = trans1.get_end(), trans1.get_start()
        n1 = int((t1_end - t1_start) / col_dt)
        ts1 = [t1_end - i * col_dt for i in range(n1 + 1)]
        ts1.reverse()
        t2_end, t2_start = trans2.get_end(), trans2.get_start()
        n2 = int((t2_end - t2_start) / col_dt)
        ts2 = [t2_end - i * col_dt for i in range(n2 + 1)]
        ts2.reverse()
        # fixme could assert that all these timestamps are the same and do the above only once outside the loop

        for t1, t2 in zip(ts1, ts2):
            x1, x2 = trans1.at_interp(t1), trans2.at_interp(t2)
            q1, q2 = extract_pose_from_state(x1), extract_pose_from_state(x2)
            a_matrix_coeff = q1[0, :2].tolist() + q1[1, :2].tolist() + q1[:2, 2].tolist()
            a_shape_tra = affine_transform(a_shape, a_matrix_coeff)
            b_matrix_coeff = q2[0, :2].tolist() + q2[1, :2].tolist() + q2[:2, 2].tolist()
            b_shape_tra = affine_transform(b_shape, b_matrix_coeff)
            if a_shape_tra.intersects(b_shape_tra):  # collision
                # many approximations here,
                # see `collision_resolution` in the simulator for a proper collision resolution
                impact_normal, impact_point = compute_impact_geometry(a_shape_tra, b_shape_tra)

                a_loc = [
                    _locations_from_impact_point(x1, impact_point),
                ]
                b_loc = [
                    _locations_from_impact_point(x2, impact_point),
                ]
                a_vel, b_vel = _approx_velocity(x1), _approx_velocity(x2)
                rel_velocity_atP = a_vel - b_vel
                a_omega, b_omega = 0, 0

                r_ap = np.array(impact_point.coords[0]) - np.array([x1.x, x1.y])
                r_bp = np.array(impact_point.coords[0]) - np.array([x2.x, x2.y])

                p_at_fault = check_who_is_at_fault(
                    p_poses={player1: q1, player2: q2}, impact_point=impact_point, lanelet_network=lanelet_network
                )
                j_n = compute_impulse_response(
                    n=impact_normal, vel_ab=rel_velocity_atP, r_ap=r_ap, r_bp=r_bp, a_geom=g1, b_geom=g2
                )
                # Apply impulse to models
                a_vel_after = velocity_after_collision(impact_normal, a_vel, g1.m, j_n)
                b_vel_after = velocity_after_collision(-impact_normal, b_vel, g2.m, j_n)
                a_omega_after = rot_velocity_after_collision(r_ap, impact_normal, np.array([0, 0, a_omega]), g1.Iz, j_n)
                b_omega_after = rot_velocity_after_collision(
                    r_bp, -impact_normal, np.array([0, 0, b_omega]), g2.Iz, j_n
                )

                # Log reports
                a_kenergy_delta = kinetic_energy(a_vel_after, g1.m) - kinetic_energy(a_vel, g1.m)
                b_kenergy_delta = kinetic_energy(b_vel_after, g2.m) - kinetic_energy(b_vel, g2.m)
                a_report = CollisionReportPlayer(
                    locations=list(zip(a_loc, impact_point.buffer(0.5))),
                    at_fault=p_at_fault[player1],
                    footprint=a_shape,
                    velocity=(a_vel, a_omega),
                    velocity_after=(a_vel_after, a_omega_after),
                    energy_delta=a_kenergy_delta,
                )
                b_report = CollisionReportPlayer(
                    locations=list(zip(b_loc, impact_point.buffer(0.5))),
                    at_fault=p_at_fault[player2],
                    footprint=b_shape,
                    velocity=(b_vel, b_omega),
                    velocity_after=(b_vel_after, b_omega_after),
                    energy_delta=b_kenergy_delta,
                )
                # todo combine report for players, what if one players collides with multiple ones in one transition?
                assert player1 not in accidents and player2 not in b_report
                accidents.update({player1: a_report, player2: b_report})
                # exit from subtransition checking
                break

    return accidents


def _approx_velocity(x: VehicleState) -> T2value:
    """This does not take into account lateral velocities"""
    v_l = np.array([x.vx, 0])
    rot: SO2value = SO2_from_angle(x.theta)
    v_g = rot @ v_l
    return v_g


def _locations_from_impact_point(state: VehicleState, impact_point: Point) -> ImpactLocation:
    direction = get_impact_point_direction(state, impact_point) % (2 * pi)

    if direction < pi / 4 or direction > pi * 7 / 4:
        return IMPACT_FRONT
    elif pi / 4 <= direction < pi * 3 / 4:
        return IMPACT_LEFT
    elif pi * 3 / 4 <= direction < pi * 5 / 4:
        return IMPACT_BACK
    elif pi * 5 / 4 <= direction < pi * 7 / 4:
        return IMPACT_RIGHT
    else:
        raise ZValueError("Unrecognised impact direction", direction=direction)
