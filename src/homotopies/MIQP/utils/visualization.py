import matplotlib.patches as patches
from matplotlib.axes import Axes
import numpy as np
from typing import Dict, Tuple
from geometry import SE2value, translation_angle_from_SE2
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.vehicle_structures import VehicleGeometry

from homotopies.MIQP.utils.intersects import traj2path, pose_from_s, get_box, get_s_max

__all__ = ["visualize_traj",
           "visualize_car",
           "visualize_pose",
           "visualize_intersect_from_s",
           "visualize_trajs_all",
           "visualize_box_2d",
           "visualize_box_3d"]


vehicle_geometry = VehicleGeometry.default_car()


def visualize_traj(traj: DgSampledSequence[SE2value], player: PlayerName, ax: Axes, color='b', plot_occupancy=True,
                   is_ref=False):
    w = vehicle_geometry.w_half
    path = np.array(traj2path(traj))  # N*2 array
    if is_ref:
        alpha = 0.3
        label = "{player}_ref".format(player=player)
    else:
        alpha = 1
        label = "{player}".format(player=player)

    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=3, linestyle='-', label=label, alpha=alpha, zorder=50)
    for idx in range(path.shape[0] - 1):
        p1 = path[idx, :]
        p2 = path[idx + 1, :]
        if plot_occupancy:
            slope_n = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + np.pi / 2
            n = np.array([w * np.cos(slope_n), w * np.sin(slope_n)])
            ax.plot([p1[0] + n[0], p2[0] + n[0]], [p1[1] + n[1], p2[1] + n[1]], color=color, linestyle='-', alpha=alpha,
                    zorder=50)
            ax.plot([p1[0] - n[0], p2[0] - n[0]], [p1[1] - n[1], p2[1] - n[1]], color=color, linestyle='-', alpha=alpha,
                    zorder=50)
    ax.legend(loc="upper left")


def visualize_car(pose: SE2value, ax: Axes, color='b', is_ref=False):
    w_half = vehicle_geometry.w_half
    l_r = vehicle_geometry.lr
    l_f = vehicle_geometry.lf
    outline = np.array([[-l_r, l_f, l_f, -l_r, -l_r],
                        [w_half, w_half, -w_half, -w_half, w_half]])
    points = np.row_stack([outline, np.ones(outline.shape[1])])
    gk = pose @ points
    if is_ref:
        alpha = 0.3
    else:
        alpha = 1
    car = patches.Polygon(gk[0:2, :].T,
                          color=color,
                          zorder=51,
                          alpha=alpha)

    # car.set_alpha(alpha)
    ax.add_patch(car)


def visualize_pose(pose: SE2value, ax: Axes):
    t, theta = translation_angle_from_SE2(pose)
    dx = 3 * np.cos(theta)
    dy = 3 * np.sin(theta)
    ax.arrow(x=t[0], y=t[1], dx=dx, dy=dy, width=.2, color='r', zorder=52)


def visualize_intersect_from_w(intersect: Tuple[float, float], ax: Axes):
    ax.plot(intersect[0], intersect[1], 'r*')


def visualize_intersect_from_s(traj: DgSampledSequence[SE2value], intersect: float, ax: Axes):
    pose = pose_from_s(traj, intersect)
    t, _ = translation_angle_from_SE2(pose)
    ax.plot(t[0], t[1], 'r*')


def visualize_trajs_all(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                        intersects: Dict[PlayerName, Dict[PlayerName, float]],
                        ax_traj: Axes,
                        color: Dict[PlayerName, str]):
    for player1 in trajs.keys():
        visualize_traj(trajs[player1], player1, ax_traj, color[player1])
        for player2 in intersects[player1].keys():
            visualize_intersect_from_s(trajs[player1], intersects[player1][player2], ax_traj)
            pose1 = pose_from_s(trajs[player1], intersects[player1][player2])
            visualize_pose(pose1, ax_traj)


def visualize_box_2d(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                     intersects: Dict[PlayerName, Dict[PlayerName, SE2value]],
                     player1: PlayerName,
                     player2: PlayerName,
                     ax: Axes,
                     box_buffer: float = 1.):
    s_max = get_s_max(trajs)
    ax.set_xlim(0, s_max[player1])
    ax.set_ylim(0, s_max[player2])
    center, w_s12, w_s21 = get_box(trajs, intersects, player1, player2, box_buffer)
    ax.plot(center[0], center[1], 'r+')
    rect_buffered = patches.Rectangle((center[0] - w_s12 / 2, center[1] - w_s21 / 2), w_s12, w_s21, linewidth=1)
    ax.add_patch(rect_buffered)
    w_s12_init = w_s12 / box_buffer
    w_s21_init = w_s21 / box_buffer
    rect_init = patches.Rectangle((center[0] - w_s12_init / 2, center[1] - w_s21_init / 2), w_s12_init, w_s21_init,
                                  linewidth=1, edgecolor='r')
    ax.add_patch(rect_init)
    ax.set_xlabel(player1)
    ax.set_ylabel(player2)


def visualize_box_3d(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                     intersects: Dict[PlayerName, Dict[PlayerName, SE2value]],
                     player1: PlayerName,
                     player2: PlayerName,
                     player3: PlayerName,
                     ax: Axes):
    s_max = get_s_max(trajs)
    x, y, z = np.indices((int(s_max[player1]), int(s_max[player2]), int(s_max[player3])))
    center12, w_s12, w_s21 = get_box(trajs, intersects, player1, player2)
    center13, w_s13, w_s31 = get_box(trajs, intersects, player1, player3)
    center23, w_s23, w_s32 = get_box(trajs, intersects, player2, player3)

    cube1 = (center12[0] - w_s12 / 2 < x) & (x < center12[0] + w_s12 / 2) & \
            (center12[1] - w_s21 / 2 < y) & (y < center12[1] + w_s21 / 2) & \
            (z < s_max[player3])
    cube2 = (center13[0] - w_s13 / 2 < x) & (x < center13[0] + w_s13 / 2) & \
            (center13[1] - w_s31 / 2 < z) & (z < center13[1] + w_s31 / 2) & \
            (y < s_max[player2])
    cube3 = (center23[0] - w_s23 / 2 < y) & (y < center23[0] + w_s23 / 2) & \
            (center23[1] - w_s32 / 2 < z) & (z < center23[1] + w_s32 / 2) & \
            (x < s_max[player1])

    voxelarray = cube1 | cube2 | cube3
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[cube1] = 'lightblue'
    colors[cube2] = 'lightgreen'
    colors[cube3] = 'gray'

    ax.voxels(voxelarray, facecolors=colors)

    ax.plot_surface((center12[0] - w_s12 / 2), y[0, :, :], z[0, :, :], alpha=0.2, color='b')
    ax.plot_surface((center12[0] + w_s12 / 2), y[0, :, :], z[0, :, :], alpha=0.2, color='b')
    ax.plot_surface((center13[0] - w_s13 / 2), y[0, :, :], z[0, :, :], alpha=0.2, color='g')
    ax.plot_surface((center13[0] + w_s13 / 2), y[0, :, :], z[0, :, :], alpha=0.2, color='g')
    ax.plot_surface(x[:, 0, :], (center12[1] - w_s21 / 2), z[:, 0, :], alpha=0.2, color='b')
    ax.plot_surface(x[:, 0, :], (center12[1] + w_s21 / 2), z[:, 0, :], alpha=0.2, color='b')
    ax.plot_surface(x[:, 0, :], (center23[0] - w_s23 / 2), z[:, 0, :], alpha=0.2, color='k')
    ax.plot_surface(x[:, 0, :], (center23[0] + w_s23 / 2), z[:, 0, :], alpha=0.2, color='k')
    ax.plot_surface(x[:, :, 0], y[:, :, 0], np.ones_like(x[:, :, 0]) * (center13[1] - w_s31 / 2), alpha=0.2, color='g')
    ax.plot_surface(x[:, :, 0], y[:, :, 0], np.ones_like(x[:, :, 0]) * (center13[1] + w_s31 / 2), alpha=0.2, color='g')
    ax.plot_surface(x[:, :, 0], y[:, :, 0], np.ones_like(x[:, :, 0]) * (center23[1] - w_s32 / 2), alpha=0.2, color='k')
    ax.plot_surface(x[:, :, 0], y[:, :, 0], np.ones_like(x[:, :, 0]) * (center23[1] + w_s32 / 2), alpha=0.2, color='k')

    ax.set_xlabel(player1)
    ax.set_ylabel(player2)
    ax.set_zlabel(player3)
