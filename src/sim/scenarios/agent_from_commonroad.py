from math import atan
from typing import List

import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.scenario.obstacle import DynamicObstacle

from sim.agent import NPAgent
from sim.models import Pacejka
from sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn, VehicleParametersDyn
from sim.models.vehicle_structures import VehicleGeometry, CAR
from sim.models.vehicle_utils import VehicleParameters


def npAgent_from_dynamic_obstacle(dyn_obs: DynamicObstacle) -> (NPAgent, VehicleModelDyn):
    assert dyn_obs.obstacle_type == VehicleType.CAR

    orientations:List[float] = [dyn_obs.initial_state.orientation]
    velocities = List[float] = [dyn_obs.initial_state.velocity]
    for state in dyn_obs.prediction.trajectory.state_list:
        orientations.append(state.orientation)
        velocities.append(state.velocity)
    dtheta = np.diff(np.array(orientations)).tolist()
    l = dyn_obs.obstacle_shape.length
    delta = [atan(l*dtheta[i]/velocities[i]) for i in range(len(velocities))]

    x0 = VehicleStateDyn(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                         theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                         delta=0)

    vg = VehicleGeometry(vehicle_type=CAR, )
    vp = VehicleParametersDyn.default_car()
    model = VehicleModelDyn(x0=x0,
                            vg=vg,
                            vp=vp,
                            pacejka_front=Pacejka.default_bicycle_front(),
                            pacejka_rear=Pacejka.default_bicycle_rear())


    agent = NPAgent()
    return agent, model

