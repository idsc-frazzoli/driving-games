import numpy as np
from typing import Dict
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim.models.utils import extract_pose_from_state
from geometry import SE2value, translation_angle_from_SE2

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
        traj = state2traj(state, 15, 0.2)  # (state, prediction horizon, time step)
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
        curr_state = VehicleState(x=t[0] + dt * (v * np.cos(theta) - vy * np.sin(theta)),
                                  y=t[1] + dt * (v * np.sin(theta) + vy * np.cos(theta)),
                                  theta=theta + dt * dtheta,
                                  vx=v,
                                  delta=delta)
        curr_pose = extract_pose_from_state(curr_state)
        time += [time[i] + dt]
        traj += [curr_pose]
    return DgSampledSequence[SE2value](time, values=traj)
