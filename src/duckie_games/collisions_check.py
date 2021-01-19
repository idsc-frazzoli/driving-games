from decimal import Decimal as D
from typing import Mapping, MutableMapping
from frozendict import frozendict

from games import PlayerName
from zuper_commons.types import ZNotImplementedError
from driving_games.collisions import Collision, IMPACT_FRONT, IMPACT_SIDES

from duckie_games.structures import DuckieGeometry, DuckieState
from duckie_games.rectangle import sample_x, rectangle_from_pose, ProjectedCar


# todo adapt for Duckies
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
        pc1 = rectangle_from_pose(s1.ref, x1, g1)
        pc2 = rectangle_from_pose(s2.ref, x2, g2)

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

        c1 = Collision(i1, p1_active, energy_received_1, energy_given_1)
        c2 = Collision(i2, p2_active, energy_received_2, energy_given_2)
        return {p1: c1, p2: c2}

    return {}


def a_caused_collision_with_b(a: ProjectedCar, b: ProjectedCar):
    return any(b.rectangle.contains(_) for _ in (a.front_right, a.front_center, a.front_left))
