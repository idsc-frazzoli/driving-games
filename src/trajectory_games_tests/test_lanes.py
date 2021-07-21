import os
from os.path import join
from typing import Dict, Set

from reprep import Report
from yaml import safe_load
import numpy as np
import geometry as geo

from games import PlayerName
from trajectory_games import config_dir, TrajGameVisualization, TrajectoryWorld, VehicleGeometry
from world import load_driving_game_map, get_lane_from_node_sequence, LaneSegmentHashable


def test_lanes():

    map_name = "4way-double-intersection-only"
    lanes_file = os.path.join(config_dir, "lanes.yaml")
    with open(lanes_file) as load_file:
        config_lanes = safe_load(load_file)[map_name]

    duckie_map = load_driving_game_map(map_name)

    lanes: Dict[PlayerName, Set[LaneSegmentHashable]] = {}
    geometries: Dict[PlayerName, VehicleGeometry] = {}

    def wrap(ang: float) -> float:
        while ang > +np.pi: ang -= 2*np.pi
        while ang < -np.pi: ang += 2*np.pi
        return abs(ang)

    i = 1
    for k, l in config_lanes.items():
        lane = LaneSegmentHashable.initializor(get_lane_from_node_sequence(m=duckie_map, node_sequence=l))
        good = True
        print(f"\nLane = {k}")
        for along1 in np.linspace(0.0, lane.get_lane_length(), 500):
            beta1 = lane.beta_from_along_lane(along1)
            point1 = lane.center_point(beta1)
            p1, r1, _ = geo.translation_angle_scale_from_E2(point1)
            beta2, point2 = lane.find_along_lane_closest_point(p1)
            along2 = lane.along_lane_from_beta(beta2)
            p2, r2, _ = geo.translation_angle_scale_from_E2(point2)
            if abs(beta1 - beta2) > 0.1 or abs(along1 - along2) > 0.5 or wrap(r1-r2) > 0.1:
                print(f"Warning, points aren't equal: \n\tbeta = {beta1, beta2} "
                      f"\n\talong = {along1, along2} \n\tangle = {r1, r2}")
                good = False
                raise Exception("Discontinuities in lane")
        if good:
            print("Lane is good, no discontinuities")
        else:
            player = PlayerName(f"P_{i}")
            lanes[player] = set(lane)
            geometries[player] = VehicleGeometry(m=100.0, w=1.0, l=1.0, colour=(1, 0, 0))
            i += 1

    print(f"\n\nTotal bad lanes = {i-1}")
    world = TrajectoryWorld(map_name=map_name, geo=geometries, lanes=lanes)
    viz = TrajGameVisualization(world=world)

    r = Report("Trajectories")
    with r.plot("actions") as pylab:
        ax = pylab.gca()
        with viz.plot_arena(axis=ax):
            pass
    d = "out/tests/"
    r.to_html(join(d, "r_lanes.html"))


if __name__ == '__main__':
    test_lanes()

