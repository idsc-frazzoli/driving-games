from math import isclose
from decimal import Decimal as D
import numpy as np

import duckietown_world as dw

from driving_games.structures import SE2_disc, NO_LIGHTS

from duckie_games.structures import DuckieState
from duckie_games.utils import get_lane_segments, merge_lanes, interpolate_along_lane, from_SE2Transform_to_SE2_disc


def test_duckie_state():
    """ Test the property functions of a DuckieState """

    duckie_map = dw.load_map('4way')

    # Get starting point
    lane_names = ['ls051', 'ls031', 'ls040', 'ls044', 'L7', 'ls005', 'ls017']
    lane_segments = get_lane_segments(duckie_map=duckie_map, lane_names=lane_names)
    lane = merge_lanes(lane_segments)
    ref_transform = interpolate_along_lane(lane=lane, along_lane=0)
    ref : SE2_disc
    ref = from_SE2Transform_to_SE2_disc(ref_transform)

    v = D(1.5)
    wait = D(1)
    light = NO_LIGHTS

    max_length = lane.get_lane_length()
    nb_points = 20
    x_along_lane = map(D, np.linspace(0, max_length, nb_points))
    for x in x_along_lane:
        ds: DuckieState
        ds = DuckieState(
            duckie_map=duckie_map,
            ref=ref,
            lane=lane,
            x=x,
            v=v,
            wait=wait,
            light=light
        )

        # Compute the poses using the duckietown world module for comparison
        test_abs_x = interpolate_along_lane(lane, float(x))
        test_abs_x_SE2_disc = from_SE2Transform_to_SE2_disc(test_abs_x)

        test_ref_x = dw.SE2Transform.from_SE2(
            dw.relative_pose(
                base=ref_transform.as_SE2(), pose=test_abs_x.as_SE2()
            )
        )

        test_ref_x_SE2_disc = from_SE2Transform_to_SE2_disc(test_ref_x)

        # Get the poses computed by the DuckieState
        abs_x: SE2_disc
        ref_x: SE2_disc
        abs_x = ds.abs_pose
        ref_x = ds.ref_pose

        msg = f"Absolute Pose is not correct: {abs_x} is not ref {test_abs_x_SE2_disc}"
        assert all(map(isclose, test_abs_x_SE2_disc, abs_x)), msg

        msg = f"Relative pose is not correct: {ref_x} is not ref {test_ref_x_SE2_disc}"
        assert all(map(isclose, test_ref_x_SE2_disc, ref_x)), msg
