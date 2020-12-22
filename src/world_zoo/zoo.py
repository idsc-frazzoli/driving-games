import os
from typing import Dict

from world.structures import load_world, World

__all__ = ["world_zoo"]

module_path = os.path.dirname(__file__)
creation_folder_path_rel = "creation_files"
creation_folder_path = os.path.join(module_path, creation_folder_path_rel)


def highway_10_lanes() -> World:
    name = "10 Lane Highway"
    png_fname = "highway-10-lanes.png"
    yaml_fname = "highway-10-lanes.yml"
    pixels_of_lane = 100  # pixel length of reference lane (first lane in yml file)
    png_path = os.path.join(creation_folder_path, png_fname)
    lane_path = os.path.join(creation_folder_path, yaml_fname)
    return load_world(
        name=name,
        background_path=png_path,
        control_points_path=lane_path,
        ref_lane_px=pixels_of_lane
    )

def intersection() -> World:
    # todo change png and yml file
    name = "Intersection"
    png_fname = "intersection.png"
    yaml_fname = "intersection.yml"
    pixels_of_lane = 100  # pixel length of reference lane (first lane in yml file)
    png_path = os.path.join(creation_folder_path, png_fname)
    lane_path = os.path.join(creation_folder_path, yaml_fname)
    return load_world(
        name=name,
        background_path=png_path,
        control_points_path=lane_path,
        ref_lane_px=pixels_of_lane
    )

loaders = [highway_10_lanes, intersection]

world_zoo : Dict[str, World] = {}
for world_loader in loaders:
    world = world_loader()
    world_name = world.name
    world_zoo.update({
        world_name : world
    })