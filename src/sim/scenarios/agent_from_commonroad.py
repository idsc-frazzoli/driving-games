from math import atan
from typing import List

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

from dg_commons import DgSampledSequence
from sim import SimTime
from sim.agents.agent import NPAgent
from sim.models import Pacejka
from sim.models.vehicle import VehicleCommands
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn, VehicleParametersDyn
from sim.models.vehicle_structures import VehicleGeometry, CAR


def npAgent_from_dynamic_obstacle(dyn_obs: DynamicObstacle, time_step: float) -> (NPAgent, VehicleModelDyn):
    """
    This function aims to create a non-playing character (fixed sequence of commands) in our simulation environment from
    a dynamic obstacle of commonroad (fixed sequence of states).
    # fixme currently only cars are supported
    # fixme this function needs to be improved...
    :param dyn_obs:
    :param time_step:
    :return:
    """
    assert dyn_obs.obstacle_type == ObstacleType.CAR

    orientations: List[float] = [dyn_obs.initial_state.orientation]
    velocities: List[float] = [dyn_obs.initial_state.velocity]
    acceleration: List[float] = [dyn_obs.initial_state.acceleration]
    for state in dyn_obs.prediction.trajectory.state_list:
        orientations.append(state.orientation)
        velocities.append(state.velocity)
        acceleration.append(state.acceleration)
    dtheta = np.diff(np.array(orientations)).tolist()
    dtheta += [dtheta[-1]]
    l = dyn_obs.obstacle_shape.length
    w_half = dyn_obs.obstacle_shape.width / 2
    delta = [atan(l * dtheta[i] / velocities[i]) for i in range(len(velocities))]
    ddelta = np.diff(np.array(delta)).tolist()
    ddelta += [ddelta[-1]]

    x0 = VehicleStateDyn(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                         theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                         delta=delta[0])
    # fixme  need to estimate mass and inertia from shape of car
    vg = VehicleGeometry(vehicle_type=CAR, w_half=w_half, m=1500.0, Iz=1000, lf=l / 2.0,
                         lr=l / 2.0, e=0.6, c_drag=0.3756, a_drag=2, color="royalblue")
    vp = VehicleParametersDyn.default_car()
    model = VehicleModelDyn(x0=x0,
                            vg=vg,
                            vp=vp,
                            pacejka_front=Pacejka.default_car_front(),
                            pacejka_rear=Pacejka.default_car_rear())

    timestamps = [SimTime(dyn_obs.initial_state.time_step * time_step), ]
    commands = [VehicleCommands(acc=acceleration[0], ddelta=ddelta[0]), ]

    for i, state in enumerate(dyn_obs.prediction.trajectory.state_list):
        timestamps.append(SimTime(state.time_step * time_step))
        commands.append(VehicleCommands(acc=acceleration[i], ddelta=ddelta[i]))
    agent = NPAgent(commands_plan=DgSampledSequence[VehicleCommands](timestamps, commands))
    return agent, model
