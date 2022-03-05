from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
from .parameters import params, x_idx, ub_idx, uc_idx, player_idx
from dg_commons import PlayerName, DgSampledSequence
from homotopies.MILP.utils.intersects import pose_from_s
from geometry import SE2value


__all__ = ["visualize_s_traj",
           "visualize_inputs",
           "visualize_ds",
           "visualize_solvetime",
           "visualize_map",
           "s2traj"]


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
    ax = plt.gca()
    if scenario is not None:
        rnd = MPRenderer()
        scenario.lanelet_network.draw(rnd, draw_params={"traffic_light": {"draw_traffic_lights": False}})
        rnd.render()
        ax.set_xlim(xmax=20, xmin=-50)
        ax.set_ylim(ymax=0, ymin=-75)
    ax.set_aspect("equal")
    return fig
