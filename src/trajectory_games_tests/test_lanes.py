import os
from yaml import safe_load
import numpy as np
import geometry as geo

from trajectory_games import config_dir
from world import load_driving_game_map, get_lane_from_node_sequence


def test_lanes():

    map_name = "4way-double-intersection-only"
    lanes_file = os.path.join(config_dir, "lanes.yaml")
    with open(lanes_file) as load_file:
        config_lanes = safe_load(load_file)[map_name]

    duckie_map = load_driving_game_map(map_name)

    for k, l in config_lanes.items():
        lane = get_lane_from_node_sequence(m=duckie_map, node_sequence=l)
        print(f"\nLane = {k}")
        for along1 in np.linspace(0.0, lane.get_lane_length(), 100):
            beta1 = lane.beta_from_along_lane(along1)
            point1 = lane.center_point(beta1)
            p1, r1, _ = geo.translation_angle_scale_from_E2(point1)
            beta2, point2 = lane.find_along_lane_closest_point(p1)
            along2 = lane.along_lane_from_beta(beta2)
            p2, r2, _ = geo.translation_angle_scale_from_E2(point2)
            if abs(beta1 - beta2) > 1e-1 or abs(along1 - along2) > 1e-1 or abs(r1-r2) > 1e-1:
                print(f"Warning, points aren't equal: \n\tbeta = {beta1, beta2} "
                      f"\n\talong = {along1, along2} \n\tangle = {r1, r2}")
                break
