import numpy as np
from typing import Dict, List, Tuple, Optional
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.utils import extract_pose_from_state
from dg_commons.maps.lanes import DgLanelet
from geometry import SE2value, translation_angle_from_SE2, SE2_from_xytheta
from dg_commons.sim.scenarios import load_commonroad_scenario

vehicle_params = VehicleParameters.default_car()
vehicle_geometry = VehicleGeometry.default_car()


def predict(obs: Dict[PlayerName, VehicleState]) -> Dict[PlayerName, DgSampledSequence[SE2value]]:
    """
    input:
    -obs: maps of agents to their current states
    output:
    maps of agents to their predicted trajectories(Sampled sequence of SE2 poses)
    """
    trajs = {}
    for player in obs.keys():
        state = obs[player]
        traj = state2traj(state, 15, 0.3)  # (state, prediction horizon, time step)
        trajs[player] = traj
    return trajs


def state2traj(state: VehicleState, horizon: float, dt: float) -> DgSampledSequence[SE2value]:
    """
    This function takes the current state of an object as input and outputs the predicted trajectory
    assuming constant vx and delta(no control inputs)
    input:
    -state: VehicleState
    -horizon: duration of the predicted trajectory
    -dt: time step used for prediction
    output:
    -traj: Sampled sequence of SE2 poses
    """
    n = int(horizon / dt)
    curr_pose = extract_pose_from_state(state)
    traj = [curr_pose]
    time = [0]
    v = state.vx
    delta = state.delta
    for i in range(n - 1):
        t, theta = translation_angle_from_SE2(traj[i])
        dtheta = v * np.tan(delta) / vehicle_geometry.length
        vy = dtheta * vehicle_geometry.lr
        curr_xytheta = [
            t[0] + dt * (v * np.cos(theta) - vy * np.sin(theta)),
            t[1] + dt * (v * np.sin(theta) + vy * np.cos(theta)),
            theta + dt * dtheta,
        ]
        curr_pose = SE2_from_xytheta(curr_xytheta)
        time += [time[i] + dt]
        traj += [curr_pose]
    return DgSampledSequence[SE2value](time, values=traj)


def traj2path(traj: DgSampledSequence[SE2value]) -> List[Tuple[float, float]]:
    """this function extracts states x,y from sampled sequence"""
    poses = traj.values
    path = []
    for i in range(len(poses)):
        t, _ = translation_angle_from_SE2(poses[i])
        path += [[t[0], t[1]]]
    return path


def traj2lane(traj: DgSampledSequence[SE2value]) -> DgLanelet:
    """
    create DgLanelet for better visualization of occupancy(not used yet)
    """
    w = vehicle_geometry.w_half
    center_vertices = np.array(traj2path(traj))
    left_vertices = np.zeros_like(center_vertices)
    right_vertices = np.zeros_like(center_vertices)
    for idx in range(len(center_vertices) - 1):
        p1 = center_vertices[idx, :]
        p2 = center_vertices[idx + 1, :]
        slope_n = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + np.pi / 2
        n = np.array([w * np.cos(slope_n), w * np.sin(slope_n)])
        if idx == 0:
            left_vertices[idx, :] = np.array([p1[0] + n[0], p1[1] + n[1]])
            right_vertices[idx, :] = np.array([p1[0] - n[0], p1[1] - n[1]])
        left_vertices[idx + 1, :] = np.array([p2[0] + n[0], p2[1] + n[1]])
        right_vertices[idx + 1, :] = np.array([p2[0] - n[0], p2[1] - n[1]])
    lane = DgLanelet.from_vertices(left_vertices, right_vertices, center_vertices)
    return lane


def traj_from_commonroad(scenario_name, scenario_dir, obstacle_id, offset: Optional[Tuple[float, float]] = None):
    """extract complex trajectory from commonroad dynamic obstacles"""
    scenario, _ = load_commonroad_scenario(scenario_name, scenario_dir)
    if offset is not None:
        offset = np.array(offset)
    else:
        offset = np.zeros(2)
    poses = []
    timestamp = []
    for obs in scenario.dynamic_obstacles:
        if obstacle_id == obs.obstacle_id:
            states = obs.prediction.trajectory.state_list
            for state in states:
                pos = state.position + offset
                new_pose = pos.tolist() + [state.orientation]
                poses += [SE2_from_xytheta(new_pose)]
                timestamp += [scenario.dt * (state.time_step - 1)]
    return DgSampledSequence[SE2value](timestamp, values=poses)
