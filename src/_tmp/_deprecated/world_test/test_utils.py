import numpy as np
from decimal import Decimal as D
from math import isclose
from copy import deepcopy

import geometry as geo

import duckietown_world as dw
from duckietown_world.svg_drawing.ipython_utils import ipython_draw_html

from _tmp._deprecated.world.utils import (
    from_SE2_disc_to_SE2Transform,
    from_SE2Transform_to_SE2_disc,
    interpolate_n_points,
    interpolate_along_lane_n_points,
    merge_lanes,
    get_lane_segments,
    LaneSegmentHashable,
    get_lane_from_node_sequence,
)
import pickle


def test_transformations():
    """Tests the conversion from SE2_discs to SE2Transforms and vice versa"""

    t_ref = [2, 4]
    theta_rad_ref = np.pi / 3
    theta_deg_ref = np.rad2deg(theta_rad_ref)

    q_SE2_ref = geo.SE2_from_translation_angle(t_ref, theta_rad_ref)
    q_SE2Transform_ref = dw.SE2Transform.from_SE2(q_SE2_ref)
    x_ref, y_ref = t_ref
    q_SE2_disc_ref = (D(x_ref), D(y_ref), D(theta_deg_ref))  # SE2_disc

    q_SE2_disc = from_SE2Transform_to_SE2_disc(q_SE2Transform_ref)

    q_SE2Transform = from_SE2_disc_to_SE2Transform(q_SE2_disc_ref)

    assert all(map(isclose, q_SE2_disc, q_SE2_disc_ref)), f"SE2_disc {q_SE2_disc} is not equal ref {q_SE2_disc_ref}"

    statement = isclose(q_SE2Transform.theta, q_SE2Transform_ref.theta) and all(
        map(isclose, q_SE2Transform.p, q_SE2Transform_ref.p)
    )
    assert statement, f"SE2transform {q_SE2Transform} is not equal ref {q_SE2Transform_ref}"


def test_lane_extracting_merging():
    """Test lane extraction from a duckietown map and their merging"""

    d = "out/"
    duckie_map = dw.load_map("4way")
    lane_names = ["ls051", "ls031", "ls040", "ls044", "L7", "ls005", "ls017"]
    lane_segments = get_lane_segments(duckie_map=duckie_map, lane_names=lane_names)
    merged_lane = merge_lanes(lane_segments)

    # Check if both lanes have same length
    sum_lane_lengths = sum([ln.get_lane_length() for ln in lane_segments])
    merged_lane_length = merged_lane.get_lane_length()
    msg = f"Lanes have not the same lenght: {sum_lane_lengths} is not {merged_lane_length}"
    assert isclose(merged_lane_length, sum_lane_lengths, abs_tol=1e-5), msg

    # Draw the list of segments
    lane_segments_obj = dw.PlacedObject()
    for lane_name, lane in zip(lane_names, lane_segments):
        lane_segments_obj.set_object(lane_name, lane, ground_truth=dw.SE2Transform.identity())
    name = "lane_segments"
    outdir = d + name
    ipython_draw_html(po=lane_segments_obj, outdir=outdir)

    # Draw the merged lane
    merged_lane_obj = dw.PlacedObject()
    merged_lane_obj.set_object("merged_lane", merged_lane, ground_truth=dw.SE2Transform.identity())
    name = "merged_lane"
    outdir = d + name
    ipython_draw_html(po=merged_lane_obj, outdir=outdir)


def test_lane_extraction_from_node():
    """
    Visually test if the lane extraction from nodes work
    """
    d = "out/"
    duckie_map = dw.load_map("4way")
    node_sequence = ["P27", "P14", "P7", "P1", "P3"]
    lane = get_lane_from_node_sequence(m=duckie_map, node_sequence=node_sequence)
    lane_obj = dw.PlacedObject()
    lane_obj.set_object("lane", lane, ground_truth=dw.SE2Transform.identity())
    name = "lane_from_node_sequence"
    outdir = d + name
    ipython_draw_html(po=lane_obj, outdir=outdir)


def test_interpolation():
    """Test for the interpolation functions"""

    d = "out/"
    duckie_map = dw.load_map("4way")
    lane_names = ["ls051", "ls031", "ls040", "ls044", "L7", "ls005", "ls017"]
    lane_segments = get_lane_segments(duckie_map=duckie_map, lane_names=lane_names)
    merged_lane = merge_lanes(lane_segments)

    nb_points = 30

    start = 0
    end = 1
    betas = np.linspace(start, end, nb_points)

    transforms_seq_0_1 = interpolate_n_points(lane=merged_lane, betas=betas)

    # Draw the interpolation sequence (0 to 1)
    duckie_map_draw = deepcopy(duckie_map)
    duckie = dw.DB18()
    timestamps = range(nb_points)
    ground_truth = dw.SampledSequence[dw.SE2Transform](timestamps, transforms_seq_0_1)
    duckie_map_draw.set_object("interpolate_0_1", duckie, ground_truth=ground_truth)
    name = "interpolated_0_1"
    outdir = d + name
    ipython_draw_html(po=duckie_map_draw, outdir=outdir)

    start = 0
    end = merged_lane.get_lane_length()
    points_along_lane = np.linspace(start, end, nb_points)

    transforms_seq_0_max_length = interpolate_along_lane_n_points(
        lane=merged_lane, positions_along_lane=points_along_lane
    )

    # Draw the interpolation sequence (0 to length lane)
    duckie_map_draw = deepcopy(duckie_map)
    duckie = dw.DB18()
    timestamps = range(nb_points)
    ground_truth = dw.SampledSequence[dw.SE2Transform](timestamps, transforms_seq_0_max_length)
    duckie_map_draw.set_object("interpolate_0_max_length", duckie, ground_truth=ground_truth)
    name = "interpolated_0_max_length"
    outdir = d + name
    ipython_draw_html(po=duckie_map_draw, outdir=outdir)


def test_hashable_lane():
    """
    Tests the hash function of the wrapper class
    """

    duckie_map = dw.load_map("4way")

    lane_names1 = ["ls051", "ls031", "ls040"]
    lane_names2 = ["ls044", "L7", "ls005", "ls017"]
    lane_segments1 = get_lane_segments(duckie_map=duckie_map, lane_names=lane_names1)
    lane1 = merge_lanes(lane_segments1)

    lane_segments2 = get_lane_segments(duckie_map=duckie_map, lane_names=lane_names2)
    lane2 = merge_lanes(lane_segments2)

    lane1_hash = LaneSegmentHashable.initializor(lane1)
    lane2_hash = LaneSegmentHashable.initializor(lane2)

    print(lane1.get_lane_length())
    print(lane2.get_lane_length())

    print(lane1_hash.get_lane_length())
    print(hash(lane1_hash))

    print(lane2_hash.get_lane_length())
    print(hash(lane2_hash))

    assert isclose(lane1.get_lane_length(), lane1_hash.get_lane_length()), "Lane 1 has not same length as hashed lane 1"
    assert isclose(lane2.get_lane_length(), lane2_hash.get_lane_length()), "Lane 2 has not same length as hashed lane 2"

    pickled_version = pickle.dumps(lane1_hash)

    lane1_hash_unpickled = pickle.loads(pickled_version)
    assert isclose(
        lane1.get_lane_length(), lane1_hash_unpickled.get_lane_length()
    ), "Lane 1 has not same length as hashed lane 1"
