# %%
from copy import deepcopy
import os
import yaml

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
module_path = 'src/world_test/'
dout = module_path + 'out/ipython_drawings/'

# %%
m = dw.load_map('4way')
# for i in range(0,3):
#     q = sample_good_starting_pose(m, only_straight=True, along_lane=0)
#     m.set_object('db18-%s'% i, dw.DB18(), ground_truth=dw.SE2Transform.from_SE2(q))
ipython_draw_html(m, outdir=dout + '4waymap')
sk = dw.get_skeleton_graph(m)
sk_root2 = sk.root2
G = sk.G
ipython_draw_html(sk_root2, outdir=dout + 'root2_car')


# %%
my_map_yaml = '4way_only.yaml'
with open(module_path + my_map_yaml) as yml_file:
    my_map_yaml_parsed = yaml.load(yml_file, Loader=yaml.SafeLoader)
mymap = dw.construct_map(my_map_yaml_parsed)
# for i in range(0,3):
#     q = sample_good_starting_pose(mymap, only_straight=True, along_lane=0)
#     mymap.set_object('db18-%s'% i, dw.DB18(), ground_truth=dw.SE2Transform.from_SE2(q))
q = sample_good_starting_pose(mymap, only_straight=True, along_lane=0.5)
ipython_draw_html(mymap, outdir=dout + my_map_yaml)
duckie = dw.DB18()
mymap.set_object('duckie', duckie, ground_truth=dw.SE2Transform.from_SE2(q))
ipython_draw_html(mymap, outdir=dout + 'mymap_duckie')
