from typing import Dict, Optional
from itertools import combinations
import os
import glob
from geometry import SE2value
import matplotlib.pyplot as plt
from PIL import Image
from reprep import Report, MIME_GIF
from zuper_commons.text import remove_escapes

from dg_commons import PlayerName, DgSampledSequence
from homotopies.MILP.utils.visualization import visualize_box_2d, visualize_car, visualize_traj
from homotopies.MILP.forces_def.visualization import *
from homotopies import logger

from .parameters import params, x_idx, ub_idx, uc_idx, player_idx


def generate_report_s_traj(X_plans, trajs, intersects, buffer=1.5):
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


def generate_report_performance(performance):
    r_performance = Report('performance')
    texts = []
    total_time = 0
    total_energy = 0
    for player in performance.keys():
        texts.append(
            f"\t{player}: \n"
            f"\ttime={performance[player][0]},\n"
            f"\tenergy={performance[player][1]}\n"
        )
        total_time += performance[player][0]
        total_energy += performance[player][1]
    texts.append(
        f"\tsum: \n"
        f"\ttime={total_time},\n"
        f"\tenergy={total_energy}\n"
    )
    text = "\n".join(texts)
    r_performance.text("Performance", remove_escapes(text))
    return r_performance


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

        ax.text(0.14,
                0.97,
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


def generate_report_solver(n_controlled, trajs, intersects, X_plans, dds_plans, solvetime, performance, homotopy, colors, scenario):
    """generate all reports for solving the path planning problem with a given homotopy class"""
    report_name = ""
    for b in homotopy.h:
        report_name += str(b)
    r = Report(report_name)
    r.add_child(get_open_loop_animation(trajs, X_plans, colors, scenario))
    r.add_child(generate_report_s_traj(X_plans, trajs, intersects))
    r.add_child(generate_report_input(dds_plans, n_controlled))
    r.add_child(generate_report_ds(X_plans, n_controlled))
    r.add_child(generate_report_solvetime(solvetime))
    r.add_child(generate_report_performance(performance))
    return r
