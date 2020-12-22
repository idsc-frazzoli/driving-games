import os

from world.structures import *

module_path = os.path.dirname(__file__)


def test_plot_world():
    d = "out/test/"
    name = "Test World"
    png_fname = "test.png"
    yaml_fname = "test.yml"
    png_path = os.path.join(module_path, png_fname)
    scale = 100/6  # [pixel/meter] Scale of background
    ref_lane_px = 100  # pixel width of reference lane (first lane in YAML-file)
    lane_path = os.path.join(module_path, yaml_fname)
    world = load_world(
        name=name,
        background_path=png_path,
        control_points_path=lane_path,
        # scale=scale,
        ref_lane_px=ref_lane_px
    )
    world.plot_world(save_png_path=d)

