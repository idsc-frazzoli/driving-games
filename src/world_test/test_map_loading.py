import os

from duckietown_world.svg_drawing.ipython_utils import ipython_draw_html

from world.map_loading import load_driving_game_map


d = "out/"


def test_new_maps():

    map_name = "4way-double"
    driving_game_map = load_driving_game_map(map_name)

    outdir = os.path.join(d, f"ipython_draw_html/{map_name}")
    ipython_draw_html(driving_game_map, outdir=outdir)
