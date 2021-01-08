from copy import deepcopy

import numpy as np

import geometry as geo
import contracts
import duckietown_world as dw
from duckietown_world.svg_drawing.ipython_utils import ipython_draw_svg, ipython_draw_html
from duckietown_world.world_duckietown.tile_template import load_tile_types
from duckietown_world.world_duckietown.sampling_poses import sample_good_starting_pose
from zuper_typing import debug_print

contracts.disable_all()
dw.logger.setLevel(50)

DELIMITER = '\n' + '-' * 80 + '\n'
DEBUG = True


def p_debug(obj, debug: bool = True):
    if debug:
        print(debug_print(obj))
        print(DELIMITER)
    else:
        pass


def interpolate(q0, q1, alpha):
    v = geo.SE2.algebra_from_group(geo.SE2.multiply(geo.SE2.inverse(q0), q1))
    vi = v * alpha
    q = np.dot(q0, geo.SE2.group_from_algebra(vi))
    return q


class Person(dw.PlacedObject):

    def __init__(self, radius, *args, **kwargs):
        self.radius = radius
        dw.PlacedObject.__init__(self, *args, **kwargs)

    def draw_svg(self, drawing, g):
        # drawing is done using the library svgwrite
        c = drawing.circle(center=(0, 0), r=self.radius, fill='pink')
        g.add(c)
        # draws x,y axes
        dw.draw_axes(drawing, g)

    def extent_points(self):
        # set of points describing the boundary
        L = self.radius
        return [(-L, -L), (+L, +L)]


def test_dw():
    q0 = geo.SE2_from_translation_angle([0, 0], 0)
    q1 = geo.SE2_from_translation_angle([2, -2], np.deg2rad(-90))

    # create a sequence of poses
    n = 10
    seqs = []
    steps = np.linspace(0, 1, num=n)
    for alpha in steps:
        q = interpolate(q0, q1, alpha)
        seqs.append(q)

    root = dw.PlacedObject()

    timestamps = range(len(seqs))  # [0, 1, 2, ...]
    transforms = [dw.SE2Transform.from_SE2(_) for _ in seqs]
    seq_me = dw.SampledSequence[dw.SE2Transform](timestamps, transforms)

    root.set_object("me", Person(0.1), ground_truth=seq_me)

    area = dw.RectangularArea((-1, -3), (3, 1))

    ipython_draw_html(root, area=area)

    template = load_tile_types()['curve_left']

    lane_segment = deepcopy(template['curve/lane1'])

    center_points = []

    for timestamp, pose_object in seq_me:
        lane_pose = lane_segment.lane_pose_from_SE2Transform(pose_object)
        print(lane_pose.center_point)
        center_points.append(lane_pose.center_point)

    sequence = dw.SampledSequence[dw.SE2Transform](seq_me.timestamps, center_points)

    lane_segment.set_object("projection2", dw.PlacedObject(), ground_truth=sequence)
    lane_segment.set_object("me", Person(0.2), ground_truth=seq_me)

    ipython_draw_html(lane_segment)


def test_dw2():
    m = dw.load_map('4way')
    ipython_draw_html(m)

    p_debug(type(m).mro(), debug=DEBUG)  # see all superclasses of the object
    p_debug(dw.get_object_tree(m, levels=4), debug=DEBUG)
    p_debug(m.children, debug=DEBUG)
    p_debug(m.children['tilemap'].children, debug=DEBUG)

    lane_segment = m['tilemap/tile-0-1/straight/lane1']  # to get tile in a compact way
    p_debug(lane_segment, debug=DEBUG)


    tile = m.children['tilemap'].children['tile-0-0']

    lane1 = tile['curve_left/curve/lane1']
    lane2 = tile['curve_left/curve/lane2']
    ipython_draw_html(lane1)
    ipython_draw_html(lane2)

    curve = tile['curve_left/curve']
    p_debug(dw.get_object_tree(curve, attributes=True, spatial_relations=True, levels=10), debug=DEBUG)

    ipython_draw_html(curve)

    lane = tile['curve_left/curve/lane2']._copy()

    ipython_draw_html(lane)

    p_debug(lane.width, debug=DEBUG)

    p_debug(lane.control_points, debug=DEBUG)

    # simple lane following

    npoints = len(lane.control_points)
    betas = list(np.linspace(-1, npoints + 1, 20))
    p_debug(betas, debug=DEBUG)

    transforms = []
    for beta in betas:
        # call the function `center_point` to get the center point (in SE(2))
        p = lane.center_point(beta)  # if n control points beta=0 first, beta=1 second, etc.
        transform = dw.SE2Transform.from_SE2(p)
        transforms.append(transform)

    ground_truth = dw.SampledSequence[dw.SE2Transform](betas, transforms)
    lane.set_object('traveling-point', dw.PlacedObject(), ground_truth=ground_truth)

    ipython_draw_html(lane)

    q = geo.SE2_from_translation_angle([+0.05, -0.05], np.deg2rad(20))

    lane.set_object('db18-4', dw.DB18(), ground_truth=dw.SE2Transform.from_SE2(q))
    lane_pose = lane.lane_pose_from_SE2(q)

    lane.set_object('marker3', dw.PlacedObject(), ground_truth=lane_pose.center_point)

    l_c = lane.children

    l_sr = lane.spatial_relations

    ipython_draw_html(lane)

    lp_cd = lane_pose.correct_direction

    lp_i = lane_pose.inside

