import os

from world.structures import *

module_path = os.path.dirname(__file__)


def test_plot_world():
    name = "Test World"
    png_fname = "test.png"
    yaml_fname = "test.yml"
    png_path = os.path.join(module_path, png_fname)
    scale = 20
    lane_path = os.path.join(module_path, yaml_fname)
    world = load_world(
        name=name,
        background_path=png_path,
        control_points_path=lane_path,
        scale=scale
    )
    world.plot_world()

