import matplotlib.patches as patches
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
from dg_commons.sim.scenarios import load_commonroad_scenario
from typing import Dict, Optional
import os
import glob
import numpy as np
from .parameters import params, x_idx, ub_idx, uc_idx, player_idx
from dg_commons import PlayerName, DgSampledSequence
from homotopies.MILP.utils.intersects import pose_from_s, traj2path
from homotopies.MILP.utils.visualization import *
from homotopies.MILP.forces_def.visualization import *
from geometry import SE2value
from PIL import Image
from reprep import Report, MIME_GIF
from homotopies import logger
from itertools import combinations


def generate_report_s_traj(X_plans, trajs, intersects, buffer=1.):
    """generate report for the box and trajectory of each intersection"""
    r_s_plan = Report(nid='s_plan')
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        with r_s_plan.plot(nid='{p1}_{p2}frame'.format(p1=player1, p2=player2)) as pylab:
            ax = pylab.gca()
            if player2 in intersects[player1].keys():
                visualize_box_2d(trajs, intersects, player1, player2, ax, buffer)
            visualize_s_traj(X_plans, player1, player2, ax)
    return r_s_plan


def generate_report_input(dds_plan, n_controlled):
    """generate report for inputs of all players"""
    r_inputs = Report('inputs')
    f = r_inputs.figure(cols=n_controlled)
    for p_idx in range(n_controlled):
        with f.plot(f"plot-{p_idx+1}") as pylab:
            ax = pylab.gca()
            visualize_inputs(dds_plan, p_idx, ax)
    return r_inputs


def generate_report_ds(X_plan, n_controlled):
    """generate report for state ds of all players"""
    r_inputs = Report('dS')
    f = r_inputs.figure(cols=n_controlled)
    for p_idx in range(n_controlled):
        with f.plot(f"plot-{p_idx+1}") as pylab:
            ax = pylab.gca()
            visualize_ds(X_plan, p_idx, ax)
    return r_inputs


def generate_report_solvetime(solvetime):
    """generate report of solvetime"""
    r_time = Report('solvetime')
    f = r_time.figure()
    with f.plot("solvetime") as pylab:
        ax = pylab.gca()
        visualize_solvetime(solvetime, ax)
    return r_time


def get_open_loop_animation(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                            X_plans,
                            colors: Dict[PlayerName, str],
                            scenario=None) -> Report:
    """create animation"""
    sim_step = X_plans.shape[2]

    tmp_folder = "images"
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    for t_idx in range(sim_step):
        curr_t = t_idx*params.dt
        logger.info(f"plotting t = {curr_t:.2f}")
        fig = visualize_map(scenario)
        ax = plt.gca()
        for player in trajs.keys():
            # plot reference path and vehicle pose
            visualize_traj(trajs[player], player, ax, color=colors[player], plot_occupancy=False, is_ref=True)
            visualize_car(pose=trajs[player].at_or_previous(t_idx*params.dt), ax=ax, color=colors[player], is_ref=True)

            # plot actual pose and predicted poses in the next N stages
            s_idx = params.n_states*player_idx[player] + x_idx.S - params.n_cinputs
            traj_plan = s2traj(X_plans[s_idx, :, t_idx], trajs[player])
            visualize_traj(traj_plan, player, ax, color='r', plot_occupancy=False)
            visualize_car(pose=traj_plan.at(traj_plan.get_start()), ax=ax, color=colors[player], is_ref=False)

            # fig = plot_traj(trajs[player], name='{player}_ref'.format(player=player), color=colors[player], opacity=0.2, fig=fig)
            # fig = plot_car(pose=trajs[player].at_or_previous(k_pred*params.dt), fig=fig, player=player, colors=colors, is_ref=True)  # reference pose without external
            # fig = plot_traj(traj_plan, name='{player}_plan'.format(player=player), fig=fig)
            # fig = plot_car(pose=traj_plan.at(traj_plan.get_start()), fig=fig, player=player, colors=colors)
            # fig.update_xaxes(
            #     scaleanchor="y",
            #     scaleratio=1,
            # )
        # fig.write_image(os.path.join(tmp_folder, f"fig{k_pred:05d}.png"))
        ax.text(0.02,
                0.96,
                f"t = {curr_t:.1f}s",
                transform=ax.transAxes,
                bbox=dict(facecolor="lightgreen", alpha=0.5),
                zorder=50
                )
        fig.savefig(os.path.join(tmp_folder, f"fig{t_idx:05d}.png"), dpi=300)
        plt.close(fig)

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(tmp_folder + "/*.png"))]
    r = Report("OpenLoopAnimation")
    with r.data_file(f"test", MIME_GIF) as f:
        duration = int(params.dt * 1e3)
        img.save(f,
                 save_all=True,
                 append_images=imgs,
                 optimize=False,
                 duration=duration,
                 loop=0)

    # clean up
    for filePath in glob.glob(tmp_folder + "/*.png"):
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file")
    return r
