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
from geometry import SE2value
from plotly.graph_objs import Figure
from PIL import Image
import plotly.graph_objects as go
from reprep import Report, MIME_GIF
from homotopies import logger
from itertools import combinations


def visualize_s_traj(X_plans, player1, player2, ax):
    """plot trajectory in s frame"""
    p1s_idx = player_idx[player1] * params.n_states + x_idx.S - params.n_cinputs
    p2s_idx = player_idx[player2] * params.n_states + x_idx.S - params.n_cinputs
    ax.plot(X_plans[p1s_idx, 0, :], X_plans[p2s_idx, 0, :], 'bo-', markersize=3)  # actual states


def visualize_inputs(dds_plan, p_idx, ax):
    """plot input at all simulation time step of player p_idx"""
    sim_time = dds_plan.shape[2]
    ax.axhline(y=params.vehicle_params.acc_limits[0], c="red", zorder=0)
    ax.axhline(y=params.vehicle_params.acc_limits[1], c="red", zorder=0)
    dds_idx = params.n_cinputs * p_idx + params.uc_idx.ddS
    ax.step(range(0, sim_time), dds_plan[dds_idx, 0, :], where='post')
    ax.set_title('input: ddS')
    ax.set_xlim(0, sim_time)
    ax.set_ylim(1.1 * params.vehicle_params.acc_limits[0], 1.1 * params.vehicle_params.acc_limits[1])
    ax.grid()


def visualize_ds(X_plan, p_idx, ax):
    """plot state ds at all simulation time step of player p_idx"""
    sim_time = X_plan.shape[2]
    ax.axhline(y=params.vehicle_params.vx_limits[0], c="red", zorder=0)
    ax.axhline(y=params.vehicle_params.vx_limits[1], c="red", zorder=0)
    ds_idx = params.n_states * p_idx + params.x_idx.dS - params.n_cinputs
    ax.step(range(0, sim_time), X_plan[ds_idx, 0, 0:sim_time], where='post')
    ax.set_title('state: dS')
    ax.set_xlim(0, sim_time)
    ax.set_ylim(1.1 * params.vehicle_params.vx_limits[0], 1.1 * params.vehicle_params.vx_limits[1])
    ax.grid()


def visualize_solvetime(solvetime, ax):
    """plot solvetime at all simulation time step"""
    sim_time = solvetime.shape[0]
    ax.plot(range(sim_time), solvetime)


def s2traj(s_plan, traj):
    """reproject s in curvilinear frame to poses in world frame"""
    dt = params.dt
    curr_time = 0
    timestamps = []
    poses = []
    for s in s_plan:
        poses += [pose_from_s(traj, s)]
        timestamps += [curr_time]
        curr_time += dt
    return DgSampledSequence[SE2value](values=poses, timestamps=timestamps)


def visualize_map(scenario):
    """plot commonroad map"""
    fig = plt.figure(figsize=(25, 10))
    fig.set_tight_layout(True)
    if scenario is not None:
        rnd = MPRenderer()
        scenario.lanelet_network.draw(rnd, draw_params={"traffic_light": {"draw_traffic_lights": False}})
        rnd.render()
    ax = plt.gca()
    ax.set_xlim(xmax=20, xmin=-50)
    ax.set_ylim(ymax=0, ymin=-75)
    ax.set_aspect("equal")
    return fig


# def plot_car(pose: SE2value, fig: Figure, player: PlayerName, colors: Dict[PlayerName, str], is_ref: bool=False) -> Figure:
#     """plot the vehicle with plotly, not used"""
#     w_half = params.vehicle_geometry.w_half
#     l_r = params.vehicle_geometry.lr
#     l_f = params.vehicle_geometry.lf
#     outline = np.array([[-l_r, l_f, l_f, -l_r, -l_r],
#                         [w_half, w_half, -w_half, -w_half, w_half]])
#     points = np.row_stack([outline, np.ones(outline.shape[1])])
#     gk = pose @ points
#     if is_ref:
#         opacity = 0.2
#         name="{player}_ref".format(player=player)
#     else:
#         opacity = 1.
#         name="{player}".format(player=player)
#     fig.add_trace(
#         go.Scatter(
#             x=gk[0, :],
#             y=gk[1, :],
#             line=dict(color=colors[player], width=1),
#             opacity=opacity,
#             fill="toself",
#             fillcolor=colors[player],
#             mode="lines",
#             name=name,
#         )
#     )
#     return fig


# def plot_traj(traj: DgSampledSequence[SE2value],
#               color='red',
#               name=None,
#               opacity=1.,
#               fig: Optional[Figure] = None) -> Figure:
#     """plot trajectory with plotly, not used"""
#     if fig is None:
#         fig = go.Figure()
#     path = traj2path(traj)
#     path = np.array(path)
#     fig.add_trace(
#         go.Scatter(
#             x=path[:, 0],
#             y=path[:, 1],
#             mode="lines",
#             line=dict(color=color),
#             name=name,
#             showlegend=True,
#             opacity=opacity
#         )
#     )
#     return fig
