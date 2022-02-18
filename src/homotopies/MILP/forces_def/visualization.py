import matplotlib.patches as patches
from typing import Dict, Optional
import os
import glob
import numpy as np
from .parameters import params, x_idx, ub_idx, uc_idx, player_idx
from dg_commons import PlayerName, DgSampledSequence
from homotopies.MILP.utils.intersects import pose_from_s, traj2path
from homotopies.MILP.utils.visualization import visualize_box_2d
from geometry import SE2value
from plotly.graph_objs import Figure
from PIL import Image
import plotly.graph_objects as go
from reprep import Report, MIME_GIF
from homotopies import logger
from itertools import combinations


def visualize_s_plan(X_plans, trajs, intersects, player1, player2, ax):
    p1s_idx = player_idx[player1] * params.n_states + x_idx.S - params.n_cinputs
    p2s_idx = player_idx[player2] * params.n_states + x_idx.S - params.n_cinputs
    ax.plot(X_plans[p1s_idx, 0, :], X_plans[p2s_idx, 0, :], 'bo-', markersize=3)  # actual states
    # for k in range(1):
    #     ax.plot(X_plans[p1s_idx, :, k], X_plans[p2s_idx, :, k], 'go-', markersize=3)


def generate_report_s_plan(X_plans, trajs, intersects, player1, player2, buffer=1.):
    r_s_plan = Report(nid='s_plan')
    for player_pair in combinations(trajs.keys(), 2):
        player1 = player_pair[0]
        player2 = player_pair[1]
        with r_s_plan.plot(nid='{p1}_{p2}frame'.format(p1=player1, p2=player2)) as pylab:
            ax = pylab.gca()
            if player2 in intersects[player1].keys():
                visualize_box_2d(trajs, intersects, player1, player2, ax, buffer)
            visualize_s_plan(X_plans, trajs, intersects, player1, player2, ax)
    return r_s_plan


def visualize_inputs(dds_plan, p_idx, sim_time, ax):
    ax.axhline(y=params.vehicle_params.acc_limits[0], c="red", zorder=0)
    ax.axhline(y=params.vehicle_params.acc_limits[1], c="red", zorder=0)
    dds_idx = params.n_cinputs * p_idx + params.uc_idx.ddS
    ax.step(range(0, sim_time), dds_plan[dds_idx, 0, 0:sim_time], where='post')
    ax.set_title('input: ddS')
    ax.set_xlim(0, sim_time)
    ax.set_ylim(1.1 * params.vehicle_params.acc_limits[0], 1.1 * params.vehicle_params.acc_limits[1])
    ax.grid()


def generate_report_input(dds_plan, n_controlled, sim_time):
    r_inputs = Report('inputs')
    f = r_inputs.figure(cols=n_controlled)
    for p_idx in range(n_controlled):
        with f.plot(f"plot-{p_idx+1}") as pylab:
            ax = pylab.gca()
            visualize_inputs(dds_plan, p_idx, sim_time, ax)
    return r_inputs


def visualize_ds(X_plan, p_idx, sim_time, ax):
    ax.axhline(y=params.vehicle_params.vx_limits[0], c="red", zorder=0)
    ax.axhline(y=params.vehicle_params.vx_limits[1], c="red", zorder=0)
    ds_idx = params.n_states * p_idx + params.x_idx.dS - params.n_cinputs
    ax.step(range(0, sim_time), X_plan[ds_idx, 0, 0:sim_time], where='post')
    ax.set_title('state: dS')
    ax.set_xlim(0, sim_time)
    ax.set_ylim(1.1 * params.vehicle_params.vx_limits[0], 1.1 * params.vehicle_params.vx_limits[1])
    ax.grid()


def generate_report_ds(X_plan, n_controlled, sim_time):
    r_inputs = Report('dS')
    f = r_inputs.figure(cols=n_controlled)
    for p_idx in range(n_controlled):
        with f.plot(f"plot-{p_idx+1}") as pylab:
            ax = pylab.gca()
            visualize_ds(X_plan, p_idx, sim_time, ax)
    return r_inputs


def generate_report_solvetime(solvetime, sim_time):
    r_time = Report('solvetime')
    f = r_time.figure()
    with f.plot("solvetime") as pylab:
        ax = pylab.gca()
        ax.plot(range(sim_time), solvetime)
    return r_time


def s2traj(s_plan, traj):
    dt = params.dt
    curr_time = 0
    timestamps = []
    poses = []
    for s in s_plan:
        poses += [pose_from_s(traj, s)]
        timestamps += [curr_time]
        curr_time += dt
    return DgSampledSequence[SE2value](values=poses, timestamps=timestamps)


def plot_traj(traj: DgSampledSequence[SE2value],
              color='red',
              name=None,
              opacity=1.,
              fig: Optional[Figure] = None) -> Figure:
    if fig is None:
        fig = go.Figure()
    path = traj2path(traj)
    path = np.array(path)
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            line=dict(color=color),
            name=name,
            showlegend=True,
            opacity=opacity
        )
    )
    return fig


def plot_car(pose: SE2value, fig: Figure, player: PlayerName, colors: Dict[PlayerName, str], is_ref: bool=False) -> Figure:
    w_half = params.vehicle_geometry.w_half
    l_r = params.vehicle_geometry.lr
    l_f = params.vehicle_geometry.lf
    outline = np.array([[-l_r, l_f, l_f, -l_r, -l_r],
                        [w_half, w_half, -w_half, -w_half, w_half]])
    points = np.row_stack([outline, np.ones(outline.shape[1])])
    gk = pose @ points
    if is_ref:
        opacity = 0.2
        name="{player}_ref".format(player=player)
    else:
        opacity = 1.
        name="{player}".format(player=player)
    fig.add_trace(
        go.Scatter(
            x=gk[0, :],
            y=gk[1, :],
            line=dict(color=colors[player], width=1),
            opacity=opacity,
            fill="toself",
            fillcolor=colors[player],
            mode="lines",
            name=name,
        )
    )
    return fig


def get_open_loop_animation(trajs: Dict[PlayerName, DgSampledSequence[SE2value]],
                            X_plans,
                            colors: Dict[PlayerName, str]) -> Report:
    n_player = len(trajs.keys())
    sim_step = X_plans.shape[2]

    tmp_folder = "images"
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    for k_pred in range(sim_step):
        logger.info(f"plotting t = {k_pred*params.dt:.2f}")
        fig = None
        for player in trajs.keys():
            fig = plot_traj(trajs[player], name='{player}_ref'.format(player=player), color=colors[player], opacity=0.2, fig=fig)
            fig = plot_car(pose=trajs[player].at_or_previous(k_pred*params.dt), fig=fig, player=player, colors=colors, is_ref=True)  # reference pose without external

            s_idx = params.n_states*player_idx[player] + x_idx.S - params.n_cinputs
            traj_plan = s2traj(X_plans[s_idx, :, k_pred], trajs[player])
            fig = plot_traj(traj_plan, name='{player}_plan'.format(player=player), fig=fig)
            fig = plot_car(pose=traj_plan.at(traj_plan.get_start()), fig=fig, player=player, colors=colors)

            fig.update_xaxes(
                scaleanchor="y",
                scaleratio=1,
            )
        fig.write_image(os.path.join(tmp_folder, f"fig{k_pred:05d}.png"))

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

