import os
from matplotlib import pyplot as plt

from world.structures import *

def test_plot_world():
    name = "Test World"
    module_path = os.path.dirname(__file__)
    png_path = os.path.join(module_path, "test.png")
    scale=1.2
    lane_path=os.path.join(module_path, "test.yml")
    world = load_world(
        name=name,
        background_path=png_path,
        control_points_path=lane_path,
        scale=scale
    )
    plt.title("Background Test")
    plt.imshow(world.background)
    plt.show()
