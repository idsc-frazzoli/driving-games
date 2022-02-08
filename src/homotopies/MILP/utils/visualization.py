import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from geometry import SE2value, translation_angle_from_SE2
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

from homotopies.MILP.utils.intersects import traj2path, pose_from_s, get_box
from homotopies.MILP.utils.intersects import compute_s_max

vehicle_geometry = VehicleGeometry.default_car()


def visualize_traj(traj: DgSampledSequence[SE2value], player: PlayerName, ax, color='b'):
    w = vehicle_geometry.w_half
    path = np.array(traj2path(traj))  # N*2 array
    ax.plot(path[:, 0], path[:, 1], color=color, marker='o', markersize=3, linestyle='-', label=player, zorder=2)
    for idx in range(path.shape[0] - 1):
        p1 = path[idx, :]
        p2 = path[idx + 1, :]
        slope_n = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + np.pi / 2
        n = np.array([w * np.cos(slope_n), w * np.sin(slope_n)])
        ax.plot([p1[0] + n[0], p2[0] + n[0]], [p1[1] + n[1], p2[1] + n[1]], color=color, linestyle='-', zorder=2)
        ax.plot([p1[0] - n[0], p2[0] - n[0]], [p1[1] - n[1], p2[1] - n[1]], color=color, linestyle='-', zorder=2)
    ax.legend()


def visualize_pose(pose: SE2value, ax):
    t, theta = translation_angle_from_SE2(pose)
    dx = 3*np.cos(theta)
    dy = 3*np.sin(theta)
    ax.arrow(x=t[0], y=t[1], dx=dx, dy=dy, width=.2, color='r', zorder=10)


def visualize_intersect_w(intersect: Tuple[float, float], ax):
    ax.plot(intersect[0], intersect[1], 'r*')


def visualize_intersect_s(traj: DgSampledSequence[SE2value], intersect: float, ax):
    pose = pose_from_s(traj, intersect)
    t, _ = translation_angle_from_SE2(pose)
    ax.plot(t[0], t[1], 'r*')


def visualize_box_2d(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                     intersects: Dict[PlayerName, Dict[PlayerName, SE2value]],
                     player1: PlayerName,
                     player2: PlayerName,
                     ax):
    center, w_s12, w_s21 = get_box(trajs, intersects, player1, player2)
    rect = patches.Rectangle((center[0] - w_s12 / 2, center[1] - w_s21 / 2), w_s12, w_s21, linewidth=1, edgecolor='r')
    ax.add_patch(rect)


def visualize_box_3d(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                     intersects: Dict[PlayerName, Dict[PlayerName, SE2value]],
                     player1: PlayerName,
                     player2: PlayerName,
                     player3: PlayerName,
                     ax):
    x, y, z = np.indices((100, 100, 100))
    center12, w_s12, w_s21 = get_box(trajs, intersects, player1, player2)
    center13, w_s13, w_s31 = get_box(trajs, intersects, player1, player3)
    center23, w_s23, w_s32 = get_box(trajs, intersects, player2, player3)

    cube1 = (center12[0]-w_s12/2 < x) & (x < center12[0]+w_s12/2) & \
            (center12[1]-w_s21/2 < y) & (y < center12[1]+w_s21/2) & \
            (z < 100)
    cube2 = (center13[0] - w_s13 / 2 < x) & (x < center13[0] + w_s13 / 2) & \
            (center13[1] - w_s31 / 2 < z) & (z < center13[1] + w_s31 / 2) & \
            (y < 100)
    cube3 = (center23[0] - w_s23 / 2 < y) & (y < center23[0] + w_s23 / 2) & \
            (center23[1] - w_s32 / 2 < z) & (z < center23[1] + w_s32 / 2) & \
            (x < 100)

    voxelarray = cube1 | cube2 | cube3
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[cube1] = 'lightblue'
    colors[cube2] = 'lightgreen'
    colors[cube3] = 'gray'

    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    ax.set_xlabel(player1)
    ax.set_ylabel(player2)
    ax.set_zlabel(player3)


def visualize_intersect_all(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                            intersects: Dict[PlayerName, Dict[PlayerName, float]],
                            player1: PlayerName,
                            player2: PlayerName,
                            ax_traj,
                            ax_box):
    path1 = traj2path(trajs[player1])
    s1_max = compute_s_max(path1)
    path2 = traj2path(trajs[player2])
    s2_max = compute_s_max(path2)
    ax_box.set_xlim([0, s1_max])
    ax_box.set_ylim([0, s2_max])
    visualize_intersect_s(trajs[player1], intersects[player1][player2], ax_traj)
    visualize_intersect_s(trajs[player2], intersects[player2][player1], ax_traj)
    pose1 = pose_from_s(trajs[player1], intersects[player1][player2])
    pose2 = pose_from_s(trajs[player2], intersects[player2][player1])
    visualize_pose(pose1, ax_traj)
    visualize_pose(pose2, ax_traj)
    visualize_box_2d(trajs, intersects, player1, player2, ax_box)
    ax_box.set_xlabel(player1)
    ax_box.set_ylabel(player2)
