from decimal import Decimal as D, localcontext
from typing import Mapping, List, Dict, Tuple
from frozendict import frozendict
import itertools
from math import isclose, sin, cos
import numpy as np

from games import PlayerName
from zuper_commons.types import ZNotImplementedError, ZValueError
from driving_games.structures import SE2_disc

from duckie_games.structures import DuckieGeometry, DuckieState
from duckie_games.collisions import (
    Collision,
    IMPACT_RIGHT,
    IMPACT_LEFT,
    IMPACT_FRONT,
    IMPACT_BACK,
    IMPACT_SIDES,
    IMPACT_NONE,
    ImpactLocation,
)

from duckie_games.rectangle import (
    sample_x,
    sample_x_speed_dep,
    ProjectedCar,
    projected_car_from_along_lane,
    two_rectangle_intersection,
    Coordinates,
    Angle_Deg
)


# todo refactor as a generic function
def collision_check(
        states: Mapping[PlayerName, DuckieState],
        geometries: Mapping[PlayerName, DuckieGeometry],
) -> Mapping[PlayerName, Collision]:
    """
    Checks for collisions in a n-player game with non-negative speeds along a lane.
    This function only checks for two player collisions and then stops the game for the players around a certain radius
    around of the accident. In urban driving this is a fair assumption, as an accident leads the other cars to stop.
    """

    players = list(states)

    collision_dict: Dict[PlayerName, Collision] = {}

    for p1, p2 in itertools.combinations(players, 2):

        if p1 in collision_dict or p2 in collision_dict:
            # Have already been in a collision or near a collision
            continue

        s1 = states[p1]
        s2 = states[p2]
        g1 = geometries[p1]
        g2 = geometries[p2]

        # samples in front and in the back of the car along the lane to account for the large time step
        dt = D(1)  # todo change for timesteps not equal 1
        n_min = 2
        n_max = 3
        x1s = sample_x_speed_dep(s1.x, s1.v, dt=dt, n_min=n_min, n_max=n_max)
        x2s = sample_x_speed_dep(s2.x, s2.v, dt=dt, n_min=n_min, n_max=n_max)

        # dt = D(0.5)
        # n = 2
        # x1s = sample_x(s1.x, s1.v, dt=dt, n=n)
        # x2s = sample_x(s2.x, s2.v, dt=dt, n=n)

        # check the sampled positions for collisions
        for x1, x2 in zip(x1s, x2s):

            # get the footprint of the car as a rectangle
            pc1 = projected_car_from_along_lane(lane=s1.lane, along_lane=x1, vg=g1)
            pc2 = projected_car_from_along_lane(lane=s2.lane, along_lane=x2, vg=g2)

            # Check if the two rectangles intersect = collision
            if not two_rectangle_intersection(pc1.rectangle, pc2.rectangle):
                # No collision
                continue

            else:
                # Collision

                # The function assumes positive speed only
                assert s1.v >= D(0) or s2.v >= D(0), (
                    f"Collision function is not suited for negative speeds ({s1.v}, {s2.v})"
                )

                # who sees the other at the instance of the collision?
                p1_sees_p2, p2_sees_p1 = who_at_fault_line_of_sight(pc1, pc2)

                # Define active as seeing the other player at collision time
                p1_active, p2_active = p1_sees_p2, p2_sees_p1

                # get the angle of collision
                p1_angle_col, p2_angle_col = get_angle_of_collision(pc1, pc2)

                col_angle_pos = p1_angle_col.copy_abs()  # convert to absolute values
                # get collision angle between 0° and 90°
                col_angle_pos_0_90 = col_angle_pos if col_angle_pos <= 90 else D(180) - col_angle_pos

                def _sin(theta): return D(sin(np.deg2rad(float(theta))))  # trigonometric functions for decimals in deg
                def _cos(theta): return D(cos(np.deg2rad(float(theta))))

                # get the location where the impact happened
                p1_impact_loc, p2_impact_loc = get_impact_location(pc1, pc2)

                # function to check if it was a collision at one of the sides
                def is_side_col(impact_loc: ImpactLocation):
                    return impact_loc == IMPACT_RIGHT or impact_loc == IMPACT_LEFT

                with localcontext() as ctx:
                    ctx.prec = 2
                    # Get the energy trasfered between the cars for different impact locations
                    if p1_impact_loc == IMPACT_FRONT and p2_impact_loc == IMPACT_FRONT:
                        # head-on collision
                        vs = s1.v * g1.mass + s2.v * g2.mass
                        energy_received_1 = vs
                        energy_received_2 = vs
                        energy_given_1 = vs
                        energy_given_2 = vs

                    elif p1_impact_loc == IMPACT_FRONT and p2_impact_loc == IMPACT_BACK:
                        # p1 drives p2 in the back
                        vs = s1.v * g1.mass - s2.v * g2.mass
                        energy_received_1 = D(0)
                        energy_received_2 = vs
                        energy_given_1 = vs
                        energy_given_2 = D(0)

                    elif p1_impact_loc == IMPACT_BACK and p2_impact_loc == IMPACT_FRONT:
                        # p2 drives p1 in the back
                        vs = s2.v * g2.mass - s1.v * g1.mass
                        energy_received_1 = vs
                        energy_received_2 = D(0)
                        energy_given_1 = D(0)
                        energy_given_2 = vs

                    elif p1_impact_loc == IMPACT_FRONT and is_side_col(p2_impact_loc):
                        # p1 drives p2 in the sides
                        vs = s1.v * g1.mass
                        energy_received_1 = D(0)
                        energy_received_2 = vs
                        energy_given_1 = vs
                        energy_given_2 = D(0)

                    elif is_side_col(p1_impact_loc) and p2_impact_loc == IMPACT_FRONT:
                        # p2 drives p1 in the sides
                        vs = s2.v * g2.mass
                        energy_received_2 = D(0)
                        energy_received_1 = vs
                        energy_given_2 = vs
                        energy_given_1 = D(0)

                    elif is_side_col(p1_impact_loc) and is_side_col(p2_impact_loc):
                        # side to side collision, take only lateral collision speed
                        half_col_angle = col_angle_pos_0_90 / D(2)

                        vs = (g1.mass * s1.v + g2.mass * s2.v) * _sin(half_col_angle)

                        energy_received_1 = vs
                        energy_received_2 = vs
                        energy_given_1 = vs
                        energy_given_2 = vs

                    elif is_side_col(p1_impact_loc) and p2_impact_loc == IMPACT_BACK:
                        # p1 drives with the sides in the back of p2, project speed of p1 along p2
                        vs_s = g1.mass * s1.v * _cos(col_angle_pos_0_90) - g2.mass * s2.v
                        vs = vs_s.copy_abs()
                        energy_received_1 = D(0)
                        energy_received_2 = vs
                        energy_given_1 = vs
                        energy_given_2 = D(0)

                    elif p1_impact_loc == IMPACT_BACK and is_side_col(p2_impact_loc):
                        # p2 drives with the sides in the back of p1, project speed of p2 along p1
                        vs_s = g2.mass * s2.v * _cos(col_angle_pos_0_90) - g1.mass * s1.v
                        vs = vs_s.copy_abs()
                        energy_received_2 = D(0)
                        energy_received_1 = vs
                        energy_given_2 = vs
                        energy_given_1 = D(0)

                    else:
                        # Case that should never happen
                        assert False, "Should not happen, check get location function"

                if energy_received_1 < 0 or energy_given_1 < 0 or energy_received_2 < 0 or energy_given_2 < 0:
                    raise ZValueError(
                        "energy is negative",
                        s1=s1,
                        s2=s2,
                        x1=x1,
                        x2=x2,
                        energy_received_1=energy_received_1,
                        energy_given_1=energy_given_1,
                        energy_received_2=energy_received_2,
                        energy_given_2=energy_given_2,
                        p1_impact_loc=p1_impact_loc,
                        p2_impact_loc=p2_impact_loc
                    )

                c1 = Collision(
                    location=p1_impact_loc,
                    angle=p1_angle_col,
                    active=p1_active,
                    energy_received=energy_received_1,
                    energy_transmitted=energy_given_1
                )
                c2 = Collision(
                    location=p2_impact_loc,
                    angle=p2_angle_col,
                    active=p2_active,
                    energy_received=energy_received_2,
                    energy_transmitted=energy_given_2
                )
                two_player_col = {p1: c1, p2: c2}
                empty_col_dict = stop_game_for_players_around(
                    colliding_players=(p1, p2),
                    states=states,
                    vg=geometries,
                    radius_factor=D(5)
                )
                collision_dict.update(empty_col_dict)
                collision_dict.update(two_player_col)

                # don't check the other positions
                break

    return collision_dict


def who_at_fault_line_of_sight(a: ProjectedCar, b: ProjectedCar) -> Tuple[bool, bool]:
    """
    Defines who is at fault, as who sees the other car at impact

                     Y
                     ^                     If there is any point
                     |/ / / / / / / /      of the contour of car 2
             x-------|-------x / / / /  <- in this region, car 1
    car 1 -> |       | / / / |/ / / /      is at fault
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
    a_orient = round(a_pose[2], 2)
    b_orient = round(b_pose[2], 2)
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
    if not isclose(angle_a, D(180)):
        angle_b = -angle_a
    else:
        angle_b = angle_a

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
        a_angle_to_y_diag <= b_angle_from_a.copy_abs() <= D(180)
        for b_angle_from_a in b_pts_angles_from_a
    ].count(True)

    # Count where the points of "a" lie in coordinate system of "b"
    b_front = [
        D(0) <= a_angle_from_b.copy_abs() <= b_angle_to_x_diag
        for a_angle_from_b in a_pts_angles_from_b
    ].count(True)

    b_left = [
        b_angle_to_x_diag < a_angle_from_b < b_angle_to_y_diag
        for a_angle_from_b in a_pts_angles_from_b
    ].count(True)

    b_right = [
        b_angle_to_x_diag < -a_angle_from_b < b_angle_to_y_diag for a_angle_from_b in a_pts_angles_from_b
    ].count(True)

    b_back = [
        b_angle_to_y_diag <= a_angle_from_b.copy_abs() <= D(180) for a_angle_from_b in a_pts_angles_from_b
    ].count(True)

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


def stop_game_for_players_around(
        colliding_players: Tuple[PlayerName, PlayerName],
        states: Mapping[PlayerName, DuckieState],
        vg: Mapping[PlayerName, DuckieGeometry],
        radius_factor: D = D(5)
) -> Mapping[PlayerName, Collision]:
    """
    Stops the game for players around a certain radius around the accident.
    radius_factor: Used in the calculation for the radius. Will multiply the sum of the two lengths
     of the colliding cars, to give the final radius.
    """

    col_player1, col_player2 = colliding_players

    radius = (vg[col_player1].length + vg[col_player1].length) * radius_factor

    pose1: SE2_disc
    pose2: SE2_disc
    pose1, pose2 = states[col_player1].abs_pose, states[col_player2].abs_pose

    center_1, center_2 = Coordinates((pose1[0], pose1[1])), Coordinates((pose2[0], pose2[1]))

    circle_center = center_1 + (center_2 - center_1) / 2

    col_dict = {}
    empty_col = Collision(
        location=IMPACT_NONE,
        angle=None,
        active=False,
        energy_received=D(0),
        energy_transmitted=D(0)
    )

    for player, state in states.items():
        if player == col_player1 or player == col_player2:
            continue

        pose: SE2_disc = state.abs_pose
        pose_center = Coordinates([pose[0], pose[1]])

        from_circle_center_to_pose = pose_center - circle_center

        dist, _ = from_circle_center_to_pose.as_polar()

        if dist <= radius:
            col_dict[player] = empty_col

    return col_dict


def collision_check_old(
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
