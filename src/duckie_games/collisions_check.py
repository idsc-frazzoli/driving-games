from decimal import Decimal as D
from typing import Mapping, MutableMapping, List, Dict, Optional, Tuple
from frozendict import frozendict
from dataclasses import dataclass
import itertools

from games import PlayerName
from zuper_commons.types import ZNotImplementedError
from driving_games.structures import SE2_disc

from duckie_games.structures import DuckieGeometry, DuckieState
from duckie_games.collisions import Collision, IMPACT_RIGHT, IMPACT_LEFT, IMPACT_FRONT, IMPACT_BACK, IMPACT_SIDES, ImpactLocation, Angle_Deg
from duckie_games.rectangle import (
    sample_x,
    projected_car_from_state,
    ProjectedCar,
    projected_car_from_along_lane,
    two_rectangle_intersection,
    Coordinates
)
from duckie_games.utils import get_pose_in_ref_frame


def collision_check(
    poses: Mapping[PlayerName, DuckieState],
    geometries: Mapping[PlayerName, DuckieGeometry],
) -> Mapping[PlayerName, Collision]:
    dt = D(0.5)
    n = 2
    if len(poses) == 1:
        return frozendict({})
    if len(poses) > 2:
        raise ZNotImplementedError(players=set(poses))

    p1, p2 = list(poses)
    s1 = poses[p1]
    s2 = poses[p2]
    g1 = geometries[p1]
    g2 = geometries[p2]

    x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
    x2s = sample_x(s2.x, s2.v, dt=dt, n=n)

    for x1, x2 in zip(x1s, x2s):
        pc1 = projected_car_from_along_lane(lane=s1.lane, along_lane=x1, vg=g1)
        pc2 = projected_car_from_along_lane(lane=s2.lane, along_lane=x2, vg=g2)

        # did p1 collide with p2?
        p1_caused = a_caused_collision_with_b(pc1, pc2)
        p2_caused = a_caused_collision_with_b(pc2, pc1)

        p1_active = p1_caused
        p2_active = p2_caused
        if p1_caused and p2_caused:
            # head-on collision
            i1 = i2 = IMPACT_FRONT
            vs = s1.v * g1.mass + s2.v * g2.mass
            energy_received_1 = vs
            energy_received_2 = vs
            energy_given_1 = vs
            energy_given_2 = vs
            pass
        elif p1_caused:
            i1 = IMPACT_FRONT
            i2 = IMPACT_SIDES
            energy_received_1 = D(0)
            energy_received_2 = s1.v * g1.mass
            energy_given_1 = s1.v * g1.mass
            energy_given_2 = D(0)
        elif p2_caused:
            i1 = IMPACT_SIDES
            i2 = IMPACT_FRONT
            energy_received_2 = D(0)
            energy_received_1 = s1.v * g1.mass
            energy_given_2 = s1.v * g1.mass
            energy_given_1 = D(0)
        else:
            continue

        c1 = Collision(i1, D(0), p1_active, energy_received_1, energy_given_1)
        c2 = Collision(i2, D(0), p2_active, energy_received_2, energy_given_2)
        return {p1: c1, p2: c2}

    return {}


def a_caused_collision_with_b(a: ProjectedCar, b: ProjectedCar):
    return any(b.rectangle.contains_point(_) for _ in (a.front_right, a.front_center, a.front_left))


# # todo adapt for Duckies then refactor as a generic function
# def collision_check_more_players(
#         states: Mapping[PlayerName, DuckieState],
#         geometries: Mapping[PlayerName, DuckieGeometry],
# ) -> Mapping[PlayerName, Collision]:
#
#     players = list(states)
#
#     for p1, p2 in itertools.combinations(players, 2):
#         s1 = states[p1]
#         s2 = states[p2]
#         g1 = geometries[p1]
#         g2 = geometries[p2]
#
#         dt = D(0.5)  # todo
#         n = 2
#         x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
#         x2s = sample_x(s2.x, s2.v, dt=dt, n=n)
#
#         for x1, x2 in zip(x1s, x2s):
#
#             pc1 = projected_car_from_along_lane(lane=s1.lane, along_lane=x1, vg=g1)
#             pc2 = projected_car_from_along_lane(lane=s2.lane, along_lane=x2, vg=g2)
#
#             if not two_rectangle_intersection(pc1.rectangle, pc2.rectangle):
#                 # No collision
#                 continue
#             else:
#                 # Collision
#
#                 # did p1 collide with p2?
#                 p1_caused, p2_caused = who_at_fault_line_of_sight(pc1, pc2)
#                 p1_active, p2_active = p1_caused, p2_caused
#
#                 p1_pose = pc1.rectangle.center_pose
#                 p2_pose = pc2.rectangle.center_pose
#
#                 p1_from_p2 = get_pose_in_ref_frame(abs_pose=p1_pose, ref=p2_pose)
#                 p2_from_p1 = get_pose_in_ref_frame(abs_pose=p2_pose, ref=p1_pose)
#
#
#                 if p1_caused and p2_caused:
#                     # head-on collision
#                     i1 = i2 = IMPACT_FRONT
#                     vs = s1.v * g1.mass + s2.v * g2.mass
#                     energy_received_1 = vs
#                     energy_received_2 = vs
#                     energy_given_1 = vs
#                     energy_given_2 = vs
#
#                 elif p1_caused:
#                     i1 = IMPACT_FRONT
#                     i2 = IMPACT_SIDES
#                     energy_received_1 = D(0)
#                     energy_received_2 = s1.v * g1.mass
#                     energy_given_1 = s1.v * g1.mass
#                     energy_given_2 = D(0)
#
#                 elif p2_caused:
#                     i1 = IMPACT_SIDES
#                     i2 = IMPACT_FRONT
#                     energy_received_2 = D(0)
#                     energy_received_1 = s1.v * g1.mass
#                     energy_given_2 = s1.v * g1.mass
#                     energy_given_1 = D(0)
#                 else:
#                     assert False, "Should not get in this case"
#
#                 # todo angle of impact
#                 c1 = Collision(D(0), p1_active, energy_received_1, energy_given_1)
#                 c2 = Collision(D(0), p2_active, energy_received_2, energy_given_2)
#                 return {p1: c1, p2: c2}
#
#     return {}


def who_at_fault_line_of_sight(a: ProjectedCar, b: ProjectedCar) -> Tuple[bool, bool]:
    """
    Defines who is at fault, as who sees the other car at impact

                     Y
                     ^                     If there is any point
                     |/ / / / / / / /      of the contour of car 2
             x-------|-------x / / / /  <- in this region, car 1
    car 1 -> |       | / / / |/ / / /      is also at fault
        -----------x--------------> X
             |       | / / / |/ / / /
             x-------|-------x / / / /
                     |/ / / / / / / /
        """

    # Get contour of both cars
    a_contour: List[Coordinates] = a.rectangle.contour
    b_contour: List[Coordinates] = b.rectangle.contour

    # Get the contour coordinates in the other cars coordinate system
    b_contour_in_a = [a.rectangle._from_abs_coord_to_rectangle(p) for p in b_contour]
    a_contour_in_b = [b.rectangle._from_abs_coord_to_rectangle(p) for p in a_contour]

    # Check if any of the points lie in the line of sight
    a_sees_b = any(coord[0] >= 0 for coord in b_contour_in_a)
    b_sees_a = any(coord[0] >= 0 for coord in a_contour_in_b)

    return a_sees_b, b_sees_a


def get_angle_of_collision(a: ProjectedCar, b: ProjectedCar) -> Tuple[Angle_Deg, Angle_Deg]:
    """
    Returns at which angle the collision happened.
    Defined as the difference between the orientations of the two cars.
    Returns the values for both cars such that the angles follow the right hand rule.


                        x_car_2
                         ^
                          \   angle_of_collision
                           \
                            \-------> x_car_1
    """

    # Get the poses of the cars
    a_pose: SE2_disc = a.rectangle.center_pose
    b_pose: SE2_disc = b.rectangle.center_pose

    # Make sure that the angles are both positive
    a_orient = a_pose[2]
    b_orient = b_pose[2]
    if a_pose[2] < 0:
        a_orient = a_orient + D(360)
    if b_pose[2] < 0:
        b_orient = b_orient + D(360)

    # get the angle between "a" and "b"
    angle_a = b_orient - a_orient

    # Make sure that he lies between -180 and 180°
    if angle_a > D(180):
        angle_a = angle_a - D(360)
    elif angle_a <= D(-180):
        angle_a = angle_a + D(360)

    # for "car b" we have to change the sign, to follow the right hand rule
    angle_b = -angle_a

    return angle_a, angle_b


def get_impact_location(a: ProjectedCar, b: ProjectedCar) -> Tuple[ImpactLocation, ImpactLocation]:
    """
    Checks the location of the impact for two cars
    E.g. for impact location for car 1. Sample points of car 2. Check how many points lie in the different cones
    generated by the diagonals of car 1. For the impact location take the side of the car that lies in the cone
    with the most points inside.

     points    x x x x    X
       of  ->   x x\ x x  ^      /
      car 2      x x\ x x |     /
                     x----|----x
                     |\   |   /|
                     | \  |  / |
                     |  \ | /  |
            Y<------------x--------------
                     |  / | \  |
                     | /  |  \ |
                     |/   |   \|
                     x----|----x
                          |
    """

    # Get the angle from the x-axis of the car to the diagonal of the first quadrant ("x") and the second one ("y")
    a_angle_diag_x, a_angle_diag_y = a.rectangle.angles_diagnoals
    a_angle_to_x_diag = a_angle_diag_x / D(2)
    a_angle_to_y_diag = a_angle_to_x_diag + a_angle_diag_y

    b_angle_diag_x, b_angle_diag_y = b.rectangle.angles_diagnoals
    b_angle_to_x_diag = b_angle_diag_x / D(2)
    b_angle_to_y_diag = b_angle_to_x_diag + b_angle_diag_y

    # Sample some points inside the cars
    a_pts_inside = a.rectangle.get_points_inside()
    b_pts_inside = b.rectangle.get_points_inside()

    # Get the coordinates of the sampled points in the coordinate systems of the two cars
    a_pts_inside_from_b = [b.rectangle._from_abs_coord_to_rectangle(_p) for _p in a_pts_inside]
    b_pts_inside_from_a = [a.rectangle._from_abs_coord_to_rectangle(_p) for _p in b_pts_inside]

    # Get the angles to the points
    a_pts_angles_from_b = [_coord.as_polar()[1] for _coord in a_pts_inside_from_b]
    b_pts_angles_from_a = [_coord.as_polar()[1] for _coord in b_pts_inside_from_a]

    # Count where the points of "b" lie in coordinate system of "a"
    a_front = [
        D(0) <= b_angle_from_a.copy_abs() <= a_angle_to_x_diag
        for b_angle_from_a in b_pts_angles_from_a
    ].count(True)

    a_left = [
        a_angle_to_x_diag < b_angle_from_a < a_angle_to_y_diag
        for b_angle_from_a in b_pts_angles_from_a
    ].count(True)

    a_right = [
        a_angle_to_x_diag < -b_angle_from_a < a_angle_to_y_diag
        for b_angle_from_a in b_pts_angles_from_a
    ].count(True)

    a_back = [
        a_angle_to_x_diag <= b_angle_from_a.copy_abs() <= D(180)
        for b_angle_from_a in b_pts_angles_from_a
    ].count(True)

    # Count where the points of "a" lie in coordinate system of "b"
    b_front = [D(0) <= a_angle_from_b.copy_abs() <= b_angle_to_x_diag for a_angle_from_b in a_pts_angles_from_b].count(True)
    b_left = [b_angle_to_x_diag < a_angle_from_b < b_angle_to_y_diag for a_angle_from_b in a_pts_angles_from_b].count(True)
    b_right = [b_angle_to_x_diag < -a_angle_from_b < b_angle_to_y_diag for a_angle_from_b in a_pts_angles_from_b].count(True)
    b_back = [b_angle_to_x_diag <= a_angle_from_b.copy_abs() <= D(180) for a_angle_from_b in a_pts_angles_from_b].count(True)

    a_impact_dict = {
        IMPACT_FRONT: a_front,
        IMPACT_LEFT: a_left,
        IMPACT_RIGHT: a_right,
        IMPACT_BACK: a_back,
    }

    b_impact_dict = {
        IMPACT_FRONT: b_front,
        IMPACT_LEFT: b_left,
        IMPACT_RIGHT: b_right,
        IMPACT_BACK: b_back,
    }

    # Impact is there where the most points lie (when there is a tie the first dict item is taken)
    a_impact = max(a_impact_dict, key=a_impact_dict.get)
    b_impact = max(b_impact_dict, key=b_impact_dict.get)

    return a_impact, b_impact
