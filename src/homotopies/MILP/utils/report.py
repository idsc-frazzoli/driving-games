from reprep import Report
from dg_commons import PlayerName, DgSampledSequence
from typing import Dict
from geometry import SE2value
from itertools import combinations
from homotopies.MILP.utils.visualization import *


def generate_report_all(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                        intersects: Dict[PlayerName, Dict[PlayerName, float]],
                        color: Dict[PlayerName, str]) -> Report:
    r = Report(nid='prediction')
    r.add_child(generate_report_trajs(trajs, intersects, color))
    r.add_child(generate_report_boxes(trajs, intersects))
    return r


def generate_report_trajs(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                          intersects: Dict[PlayerName, Dict[PlayerName, float]],
                          color: Dict[PlayerName, str]) -> Report:
    r_trajs = Report(nid='trajectories')
    with r_trajs.plot(nid='world_frame') as pylab:
        ax_traj = pylab.gca()
        visualize_trajs_all(trajs, intersects, ax_traj, color)
    return r_trajs


def generate_report_boxes(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                          intersects: Dict[PlayerName, Dict[PlayerName, float]]) -> Report:
    r_boxes = Report(nid='boxes')
    n_player = len(trajs.keys())
    n_plot = int(n_player * (n_player - 1) / 2)
    f = r_boxes.figure(cols=n_plot)
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        with f.plot(f"plot-{player1}-{player2}") as pylab:
            ax_box = pylab.gca()
            visualize_box_2d(trajs, intersects, player1, player2, ax_box)
    return r_boxes


def generate_report_3d_boxes(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                             intersects: Dict[PlayerName, Dict[PlayerName, float]],
                             player1: PlayerName,
                             player2: PlayerName,
                             player3: PlayerName) -> Report:
    # todo: don't know how to set projection=3d
    r_3d = Report(nid='3d_boxes')
    with r_3d.plot(nid='test') as pylab:
        ax = pylab.gca()
        visualize_box_3d(trajs, intersects, player1, player2, player3, ax)
    return r_3d

