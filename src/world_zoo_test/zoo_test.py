from parameterized import parameterized
from typing import List

from world.structures import World
from world_zoo.zoo import world_zoo

world_names = {
    #"10 Lane Highway",
    "Intersection"
}
worlds = [(world_zoo[name],) for name in world_names]

@parameterized(worlds)
def test_world_zoo(world: World):
    d = "out/tests/"
    world.plot_world(save_png_path=d)
