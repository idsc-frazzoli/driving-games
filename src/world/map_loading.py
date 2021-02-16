# coding=utf-8
import oyaml as yaml
import os

from duckietown_world.world_duckietown.duckietown_map import DuckietownMap
from duckietown_world.world_duckietown.old_map_format import MapFormat1Object

from duckietown_world.world_duckietown.tile import Tile
from duckietown_world.world_duckietown.tile_map import TileMap
from duckietown_world.world_duckietown.map_loading import get_object, get_transform

from duckietown_world.geo import Scale2D, SE2Transform
from duckietown_world.geo.measurements_utils import iterate_by_class

from world.tiles import load_driving_games_tile_types

__all__ = ["load_driving_game_map", "load_driving_game_map_from_yaml"]

module_path = os.path.dirname(__file__)

map_directory = os.path.join(module_path, 'maps')


def load_driving_game_map(name: str) -> DuckietownMap:
    """
    Loads a Driving Game map out of the maps folder
    """
    yml_path = os.path.join(map_directory, f"{name}.yaml")
    return load_driving_game_map_from_yaml(yml_path)


def load_driving_game_map_from_yaml(path: str) -> DuckietownMap:
    """
    Loads Driving Game Map out of a yaml file
    """
    with open(path) as yml_file:
        driving_game_yaml_parsed = yaml.load(yml_file, Loader=yaml.SafeLoader)
    driving_game_map = construct_driving_game_map(driving_game_yaml_parsed)
    return driving_game_map


def construct_driving_game_map(yaml_data: dict) -> DuckietownMap:
    """
    Function forked from of the duckietown world module
    """
    tile_size = yaml_data["tile_size"]
    dm = DuckietownMap(tile_size)
    tiles = yaml_data["tiles"]
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    A = len(tiles)
    B = len(tiles[0])
    tm = TileMap(H=B, W=A)

    templates = load_driving_games_tile_types()
    for a, row in enumerate(tiles):
        if len(row) != B:
            msg = "each row of tiles must have the same length"
            raise ValueError(msg)

        # For each tile in this row
        for b, tile in enumerate(row):
            tile = tile.strip()

            if tile == "empty":
                continue

            DEFAULT_ORIENT = "E"  # = no rotation
            if "/" in tile:
                kind, orient = tile.split("/")
                kind = kind.strip()
                orient = orient.strip()

                drivable = True

            elif "4" in tile and "double" in tile:
                kind = "4way_double"
                # angle = 2
                orient = DEFAULT_ORIENT
                drivable = True

            elif "4" in tile:
                kind = "4way"
                # angle = 2
                orient = DEFAULT_ORIENT
                drivable = True
            else:
                kind = tile
                # angle = 0
                orient = DEFAULT_ORIENT
                drivable = False

            tile = Tile(kind=kind, drivable=drivable)
            if kind in templates:
                tile.set_object(kind, templates[kind], ground_truth=SE2Transform.identity())
            else:
                pass
                # msg = 'Could not find %r in %s' % (kind, templates)
                # logger.debug(msg)

            tm.add_tile(b, (A - 1) - a, orient, tile)

    def go(obj_name0: str, desc0: MapFormat1Object):
        obj = get_object(desc0)
        transform = get_transform(desc0, tm.W, tile_size)
        dm.set_object(obj_name0, obj, ground_truth=transform)

    objects = yaml_data.get("objects", [])
    if isinstance(objects, list):
        for obj_idx, desc in enumerate(objects):
            kind = desc["kind"]
            obj_name = f"ob{obj_idx:02d}-{kind}"
            go(obj_name, desc)
    elif isinstance(objects, dict):
        for obj_name, desc in objects.items():
            go(obj_name, desc)
    else:
        raise ValueError(objects)

    for it in list(iterate_by_class(tm, Tile)):
        ob = it.object
        if "slots" in ob.children:
            slots = ob.children["slots"]
            for k, v in list(slots.children.items()):
                if not v.children:
                    slots.remove_object(k)
            if not slots.children:
                ob.remove_object("slots")

    dm.set_object("tilemap", tm, ground_truth=Scale2D(tile_size))
    return dm
